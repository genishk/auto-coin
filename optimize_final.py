"""
최종 RSI + 손절 최적화 (골든크로스 필터 적용)
- 넓은 범위 테스트
- 일봉 5년 + 4시간봉 2년 평균 수익률 기준
- 최대 ~1000개 조합
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

def calculate_indicators(df):
    """기술 지표 계산"""
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    return df

def find_buy_signals(df, rsi_oversold, rsi_exit):
    """매수 시그널 (골든크로스 필터 적용)"""
    signals = []
    in_oversold = False
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        gc = df['golden_cross'].iloc[idx]
        
        if pd.isna(rsi) or pd.isna(gc):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
        elif in_oversold and rsi >= rsi_exit:
            if gc:
                signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
            in_oversold = False
    
    return signals

def find_sell_signals(df, rsi_overbought, rsi_exit):
    """매도 시그널"""
    signals = []
    in_overbought = False
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
        elif in_overbought and rsi <= rsi_exit:
            signals.append({
                'confirm_date': df.index[idx],
                'confirm_price': df['Close'].iloc[idx]
            })
            in_overbought = False
    
    return signals

def simulate_new_strategy(df, buy_signals, sell_signals, stop_loss=-25):
    """시뮬레이션 (수익일 때만 익절)"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            avg_price = sum(p['price'] for p in positions) / len(positions)
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = current_price
            
            if current_return <= stop_loss:
                exit_reason = "손절"
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({
                    'num_buys': len(positions),
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    current_pos = len(positions) if positions else 0
    
    return trades, current_pos

def main():
    print("="*70)
    print("🔬 최종 RSI + 손절 최적화 (골든크로스 필터 적용)")
    print("="*70)
    
    # 데이터 로드
    print("\n📊 데이터 로딩 중...")
    df_1d = get_data('1d')
    df_4h = get_data('4h')
    
    df_1d = calculate_indicators(df_1d)
    df_4h = calculate_indicators(df_4h)
    
    print(f"   일봉: {len(df_1d)}개, 4시간봉: {len(df_4h)}개")
    
    # 조합 설정 (약 800개)
    # RSI 과매도: 25, 30, 35, 40 (4개)
    # RSI 매수탈출: 35, 40, 45, 50, 55 (5개) - 과매도보다 높아야 함
    # RSI 과매수: 70, 75, 80, 85 (4개)
    # RSI 매도탈출: 45, 50, 55, 60, 65 (5개) - 과매수보다 낮아야 함
    # 손절: -15, -20, -25, -30, -35 (5개)
    
    rsi_oversolds = [25, 30, 35, 40]
    rsi_buy_exits = [35, 40, 45, 50, 55]
    rsi_overboughts = [70, 75, 80, 85]
    rsi_sell_exits = [45, 50, 55, 60, 65]
    stop_losses = [-15, -20, -25, -30, -35]
    
    # 유효 조합 수 계산
    valid_count = 0
    for oversold in rsi_oversolds:
        for buy_exit in rsi_buy_exits:
            if buy_exit <= oversold:
                continue
            for overbought in rsi_overboughts:
                for sell_exit in rsi_sell_exits:
                    if sell_exit >= overbought:
                        continue
                    valid_count += len(stop_losses)
    
    print(f"\n🔄 총 {valid_count}개 유효 조합 테스트 중...")
    
    results = []
    
    with tqdm(total=valid_count, desc="테스트 중") as pbar:
        for oversold in rsi_oversolds:
            for buy_exit in rsi_buy_exits:
                if buy_exit <= oversold:
                    continue
                
                for overbought in rsi_overboughts:
                    for sell_exit in rsi_sell_exits:
                        if sell_exit >= overbought:
                            continue
                        
                        for stop_loss in stop_losses:
                            # 일봉 테스트
                            buy_1d = find_buy_signals(df_1d, oversold, buy_exit)
                            sell_1d = find_sell_signals(df_1d, overbought, sell_exit)
                            trades_1d, pos_1d = simulate_new_strategy(df_1d, buy_1d, sell_1d, stop_loss)
                            
                            # 4시간봉 테스트
                            buy_4h = find_buy_signals(df_4h, oversold, buy_exit)
                            sell_4h = find_sell_signals(df_4h, overbought, sell_exit)
                            trades_4h, pos_4h = simulate_new_strategy(df_4h, buy_4h, sell_4h, stop_loss)
                            
                            # 수익률 계산
                            ret_1d = 0
                            if trades_1d:
                                total_ret = 1.0
                                for t in trades_1d:
                                    total_ret *= (1 + t['return'] / 100)
                                ret_1d = (total_ret - 1) * 100
                            
                            ret_4h = 0
                            if trades_4h:
                                total_ret = 1.0
                                for t in trades_4h:
                                    total_ret *= (1 + t['return'] / 100)
                                ret_4h = (total_ret - 1) * 100
                            
                            avg_ret = (ret_1d + ret_4h) / 2
                            
                            results.append({
                                'oversold': oversold,
                                'buy_exit': buy_exit,
                                'overbought': overbought,
                                'sell_exit': sell_exit,
                                'stop_loss': stop_loss,
                                'ret_1d': ret_1d,
                                'ret_4h': ret_4h,
                                'avg_ret': avg_ret,
                                'trades_1d': len(trades_1d),
                                'trades_4h': len(trades_4h),
                                'pos_1d': pos_1d,
                                'pos_4h': pos_4h
                            })
                            
                            pbar.update(1)
    
    # 결과 정렬
    results.sort(key=lambda x: x['avg_ret'], reverse=True)
    
    # 현재 포지션 없는 것만 필터
    no_pos = [r for r in results if r['pos_1d'] == 0 and r['pos_4h'] == 0]
    no_pos.sort(key=lambda x: x['avg_ret'], reverse=True)
    
    # 결과 출력
    print("\n" + "="*70)
    print("📊 전체 평균 수익률 Top 10")
    print("="*70)
    
    print(f"\n{'조합':<20} {'손절':>6} {'일봉':>10} {'4시간':>10} {'평균':>10} {'포지션':>10}")
    print("-"*70)
    
    for r in results[:10]:
        combo = f"{r['oversold']}/{r['buy_exit']}/{r['overbought']}/{r['sell_exit']}"
        pos_str = f"{r['pos_1d']}/{r['pos_4h']}회"
        print(f"{combo:<20} {r['stop_loss']:>5}% {r['ret_1d']:>+9.1f}% {r['ret_4h']:>+9.1f}% {r['avg_ret']:>+9.1f}% {pos_str:>10}")
    
    # 현재 포지션 없는 것 Top 10
    print("\n" + "="*70)
    print("🛡️ 현재 포지션 없는 것 Top 10 (확정 수익률)")
    print("="*70)
    
    print(f"\n{'조합':<20} {'손절':>6} {'일봉':>10} {'4시간':>10} {'평균':>10} {'거래수':>10}")
    print("-"*70)
    
    for r in no_pos[:10]:
        combo = f"{r['oversold']}/{r['buy_exit']}/{r['overbought']}/{r['sell_exit']}"
        trades_str = f"{r['trades_1d']}/{r['trades_4h']}"
        print(f"{combo:<20} {r['stop_loss']:>5}% {r['ret_1d']:>+9.1f}% {r['ret_4h']:>+9.1f}% {r['avg_ret']:>+9.1f}% {trades_str:>10}")
    
    # 손절별 Top 1
    print("\n" + "="*70)
    print("📊 손절 라인별 최고 조합 (포지션 없는 것)")
    print("="*70)
    
    for sl in stop_losses:
        sl_results = [r for r in no_pos if r['stop_loss'] == sl]
        if sl_results:
            best = sl_results[0]
            combo = f"{best['oversold']}/{best['buy_exit']}/{best['overbought']}/{best['sell_exit']}"
            print(f"\n손절 {sl}%: {combo}")
            print(f"   일봉: {best['ret_1d']:+.1f}%, 4시간: {best['ret_4h']:+.1f}%, 평균: {best['avg_ret']:+.1f}%")
    
    # 최종 추천
    print("\n" + "="*70)
    print("🏆 최종 추천")
    print("="*70)
    
    if no_pos:
        best = no_pos[0]
        print(f"\n✅ 최적 조합: {best['oversold']}/{best['buy_exit']}/{best['overbought']}/{best['sell_exit']}")
        print(f"   손절: {best['stop_loss']}%")
        print(f"   일봉 수익률: {best['ret_1d']:+.1f}%")
        print(f"   4시간 수익률: {best['ret_4h']:+.1f}%")
        print(f"   평균 수익률: {best['avg_ret']:+.1f}%")
        print(f"   거래 횟수: 일봉 {best['trades_1d']}회, 4시간 {best['trades_4h']}회")

if __name__ == '__main__':
    main()

