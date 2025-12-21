"""
골든크로스 MA 수치 최적화
- 다양한 MA 조합으로 일봉/4시간봉 수익률 비교
"""
import sys
sys.path.insert(0, '.')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def get_data(interval='1d'):
    """데이터 가져오기"""
    ticker = 'BTC-USD'
    end_date = datetime.now()
    
    if interval == '4h':
        start_date = end_date - timedelta(days=729)
    else:
        start_date = end_date - timedelta(days=365*5)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def calculate_indicators(df, short_ma, long_ma):
    """기술 지표 계산"""
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 이동평균
    df['MA_short'] = df['Close'].rolling(window=short_ma).mean()
    df['MA_long'] = df['Close'].rolling(window=long_ma).mean()
    
    return df

def simulate_strategy(df, use_golden_cross=True,
                      rsi_oversold=35, rsi_buy_exit=40, rsi_overbought=80, rsi_sell_exit=55,
                      stop_loss_pct=-0.25):
    """전략 시뮬레이션"""
    
    trades = []
    position = None
    
    in_oversold = False
    in_overbought = False
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        ma_short = df['MA_short'].iloc[i]
        ma_long = df['MA_long'].iloc[i]
        
        if pd.isna(rsi) or pd.isna(ma_long):
            continue
        
        # 골든크로스 체크
        golden_cross_ok = ma_short > ma_long if use_golden_cross else True
        
        # 포지션 있을 때
        if position is not None:
            avg_price = position['avg_price']
            current_return = (price - avg_price) / avg_price
            
            # 손절 체크
            if current_return <= stop_loss_pct:
                trades.append({
                    'num_buys': len(position['entries']),
                    'return_pct': current_return * 100,
                    'exit_reason': '손절'
                })
                position = None
                in_oversold = False
                in_overbought = False
                continue
            
            # 매도 시그널 체크
            if rsi > rsi_overbought:
                in_overbought = True
            
            if in_overbought and rsi <= rsi_sell_exit:
                if current_return > 0:
                    trades.append({
                        'num_buys': len(position['entries']),
                        'return_pct': current_return * 100,
                        'exit_reason': '익절'
                    })
                    position = None
                in_overbought = False
                continue
        
        # 매수 시그널 체크
        if rsi < rsi_oversold:
            in_oversold = True
        
        if in_oversold and rsi >= rsi_buy_exit:
            if golden_cross_ok:
                if position is None:
                    position = {
                        'entries': [(price,)],
                        'avg_price': price
                    }
                else:
                    position['entries'].append((price,))
                    total_cost = sum(p[0] for p in position['entries'])
                    position['avg_price'] = total_cost / len(position['entries'])
            
            in_oversold = False
    
    # 현재 포지션
    current_position = None
    if position is not None:
        current_price = df['Close'].iloc[-1]
        avg_price = position['avg_price']
        current_return = (current_price - avg_price) / avg_price
        current_position = {
            'num_buys': len(position['entries']),
            'return_pct': current_return * 100
        }
    
    # 총 수익률 계산
    total_return = 1.0
    for t in trades:
        total_return *= (1 + t['return_pct'] / 100)
    total_return = (total_return - 1) * 100
    
    max_buys = max([t['num_buys'] for t in trades], default=0)
    
    return {
        'total_return': total_return,
        'num_trades': len(trades),
        'max_buys': max_buys,
        'current_position': current_position
    }

def main():
    print("="*60)
    print("🔬 골든크로스 MA 수치 최적화")
    print("="*60)
    
    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    df_1d = get_data('1d')
    df_4h = get_data('4h')
    print(f"   일봉: {len(df_1d)}개, 4시간봉: {len(df_4h)}개")
    
    # MA 조합
    short_mas = [20, 30, 40, 50, 60, 70]
    long_mas = [100, 150, 200]
    
    results = []
    
    print(f"\n🔄 {len(short_mas) * len(long_mas)}개 조합 테스트 중...")
    
    for short_ma in tqdm(short_mas, desc="테스트 중"):
        for long_ma in long_mas:
            if short_ma >= long_ma:
                continue
            
            # 일봉 테스트
            df_1d_test = calculate_indicators(df_1d.copy(), short_ma, long_ma)
            result_1d = simulate_strategy(df_1d_test, use_golden_cross=True)
            
            # 4시간봉 테스트
            df_4h_test = calculate_indicators(df_4h.copy(), short_ma, long_ma)
            result_4h = simulate_strategy(df_4h_test, use_golden_cross=True)
            
            # 현재 포지션 정보
            pos_1d = result_1d['current_position']
            pos_4h = result_4h['current_position']
            
            results.append({
                'short_ma': short_ma,
                'long_ma': long_ma,
                'return_1d': result_1d['total_return'],
                'return_4h': result_4h['total_return'],
                'avg_return': (result_1d['total_return'] + result_4h['total_return']) / 2,
                'trades_1d': result_1d['num_trades'],
                'trades_4h': result_4h['num_trades'],
                'max_buys_1d': result_1d['max_buys'],
                'max_buys_4h': result_4h['max_buys'],
                'current_buys_1d': pos_1d['num_buys'] if pos_1d else 0,
                'current_buys_4h': pos_4h['num_buys'] if pos_4h else 0,
            })
    
    # 결과 정렬 (평균 수익률 기준)
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 결과 (평균 수익률 순)")
    print("="*60)
    
    print(f"\n{'MA조합':<12} {'일봉5년':>10} {'4시간2년':>10} {'평균':>10} {'현재물타기':>12}")
    print("-"*60)
    
    for r in results[:10]:
        ma_str = f"MA{r['short_ma']}/{r['long_ma']}"
        current_str = f"{r['current_buys_1d']}/{r['current_buys_4h']}회"
        print(f"{ma_str:<12} {r['return_1d']:>+9.1f}% {r['return_4h']:>+9.1f}% {r['avg_return']:>+9.1f}% {current_str:>12}")
    
    # 현재 50/200과 비교
    print("\n" + "="*60)
    print("📊 현재 설정 (MA50/200)과 비교")
    print("="*60)
    
    current = next((r for r in results if r['short_ma'] == 50 and r['long_ma'] == 200), None)
    best = results[0]
    
    if current:
        print(f"\n현재 MA50/200:")
        print(f"   일봉 5년: {current['return_1d']:+.1f}%")
        print(f"   4시간 2년: {current['return_4h']:+.1f}%")
        print(f"   평균: {current['avg_return']:+.1f}%")
    
    print(f"\n🏆 최적 MA{best['short_ma']}/{best['long_ma']}:")
    print(f"   일봉 5년: {best['return_1d']:+.1f}%")
    print(f"   4시간 2년: {best['return_4h']:+.1f}%")
    print(f"   평균: {best['avg_return']:+.1f}%")
    
    # 필터 없는 경우와 비교
    print("\n" + "="*60)
    print("📊 골든크로스 필터 vs 필터 없음")
    print("="*60)
    
    # 필터 없이 테스트
    df_1d_no = calculate_indicators(df_1d.copy(), 50, 200)
    result_1d_no = simulate_strategy(df_1d_no, use_golden_cross=False)
    
    df_4h_no = calculate_indicators(df_4h.copy(), 50, 200)
    result_4h_no = simulate_strategy(df_4h_no, use_golden_cross=False)
    
    avg_no = (result_1d_no['total_return'] + result_4h_no['total_return']) / 2
    
    print(f"\n필터 없음 (현재 전략):")
    print(f"   일봉 5년: {result_1d_no['total_return']:+.1f}%")
    print(f"   4시간 2년: {result_4h_no['total_return']:+.1f}%")
    print(f"   평균: {avg_no:+.1f}%")
    print(f"   현재 물타기: {result_1d_no['current_position']['num_buys'] if result_1d_no['current_position'] else 0}/{result_4h_no['current_position']['num_buys'] if result_4h_no['current_position'] else 0}회")
    
    print(f"\n🏆 최적 골든크로스 MA{best['short_ma']}/{best['long_ma']}:")
    print(f"   일봉 5년: {best['return_1d']:+.1f}%")
    print(f"   4시간 2년: {best['return_4h']:+.1f}%")
    print(f"   평균: {best['avg_return']:+.1f}%")
    print(f"   현재 물타기: {best['current_buys_1d']}/{best['current_buys_4h']}회")

if __name__ == '__main__':
    main()

