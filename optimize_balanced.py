"""
균형 잡힌 RSI 기준 찾기
- 일봉 5년 + 4시간봉 2년 평균 수익률 최적화
- 약 1000개 조합
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators


def find_buy_signals(df, rsi_oversold, rsi_exit):
    signals = []
    in_oversold = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi < rsi_oversold and not in_oversold:
            in_oversold = True
        elif in_oversold and current_rsi >= rsi_exit:
            signals.append({
                'confirm_date': df.index[i],
                'confirm_price': row['Close']
            })
            in_oversold = False
    
    return signals


def find_sell_signals(df, rsi_overbought, rsi_exit):
    signals = []
    in_overbought = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi > rsi_overbought and not in_overbought:
            in_overbought = True
        elif in_overbought and current_rsi <= rsi_exit:
            signals.append({
                'confirm_date': df.index[i],
                'confirm_price': row['Close']
            })
            in_overbought = False
    
    return signals


def simulate_new_strategy(df, buy_signals, sell_signals, stop_loss=-25):
    """새 전략: 수익일 때만 매도"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df.iloc[idx]['Close']
        
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
                trades.append({'return': final_return})
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, len(positions)


def calc_return(trades):
    if not trades:
        return 0
    cumulative = 1.0
    for t in trades:
        cumulative *= (1 + t['return'] / 100)
    return (cumulative - 1) * 100


def main():
    print("=" * 80)
    print("🔬 균형 잡힌 RSI 기준 찾기 (일봉 5년 + 4시간봉 2년 평균)")
    print("=" * 80)
    print()
    
    ticker = 'BTC-USD'
    ti = TechnicalIndicators()
    
    # 4시간봉 2년 로드
    print("📊 4시간봉 2년 데이터 로딩...")
    cache_4h = DataCache(cache_dir='data/cache_4h', max_age_hours=4)
    df_4h = cache_4h.get(f'{ticker}_4h')
    
    if df_4h is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        df_4h = data[ticker]
        df_4h, _ = validate_data(df_4h, ticker)
        cache_4h.set(f'{ticker}_4h', df_4h)
    
    df_4h = ti.calculate_all(df_4h)
    print(f"   기간: {df_4h.index[0].strftime('%Y-%m-%d')} ~ {df_4h.index[-1].strftime('%Y-%m-%d')}")
    
    # 일봉 5년 로드
    print("📊 일봉 5년 데이터 로딩...")
    cache_1d = DataCache(cache_dir='data/cache', max_age_hours=24)
    df_1d = cache_1d.get(f'{ticker}_1d')
    
    if df_1d is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='5y', interval='1d')
        df_1d = data[ticker]
        df_1d, _ = validate_data(df_1d, ticker)
        cache_1d.set(f'{ticker}_1d', df_1d)
    
    df_1d = ti.calculate_all(df_1d)
    print(f"   기간: {df_1d.index[0].strftime('%Y-%m-%d')} ~ {df_1d.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    # 파라미터 범위 (약 1000개 조합)
    oversold_range = [15, 20, 25, 30, 35]              # 5개
    buy_exit_range = [30, 35, 40, 45, 50, 55, 60]      # 7개
    overbought_range = [65, 70, 75, 80, 85]            # 5개
    sell_exit_range = [30, 35, 40, 45, 50, 55]         # 6개
    
    # 5 × 7 × 5 × 6 = 1,050개
    total = len(oversold_range) * len(buy_exit_range) * len(overbought_range) * len(sell_exit_range)
    print(f"🔄 총 {total}개 조합 테스트 중...")
    print()
    
    results = []
    combinations = list(product(oversold_range, buy_exit_range, overbought_range, sell_exit_range))
    
    for oversold, buy_exit, overbought, sell_exit in tqdm(combinations, desc="최적화"):
        if buy_exit <= oversold:
            continue
        if sell_exit >= overbought:
            continue
        
        # 4시간봉 테스트
        buy_4h = find_buy_signals(df_4h, oversold, buy_exit)
        sell_4h = find_sell_signals(df_4h, overbought, sell_exit)
        trades_4h, pos_4h = simulate_new_strategy(df_4h, buy_4h, sell_4h)
        return_4h = calc_return(trades_4h)
        
        # 일봉 테스트
        buy_1d = find_buy_signals(df_1d, oversold, buy_exit)
        sell_1d = find_sell_signals(df_1d, overbought, sell_exit)
        trades_1d, pos_1d = simulate_new_strategy(df_1d, buy_1d, sell_1d)
        return_1d = calc_return(trades_1d)
        
        # 평균 수익률
        avg_return = (return_4h + return_1d) / 2
        
        # 최소 거래 횟수 필터
        if len(trades_4h) >= 3 and len(trades_1d) >= 3:
            results.append({
                'oversold': oversold,
                'buy_exit': buy_exit,
                'overbought': overbought,
                'sell_exit': sell_exit,
                'return_4h': return_4h,
                'return_1d': return_1d,
                'avg_return': avg_return,
                'trades_4h': len(trades_4h),
                'trades_1d': len(trades_1d),
                'pos_4h': pos_4h,
                'pos_1d': pos_1d
            })
    
    results_df = pd.DataFrame(results)
    
    print()
    print("=" * 100)
    print("🏆 평균 수익률 기준 Top 20")
    print("=" * 100)
    print()
    print(f"{'#':<3} {'과매도':<7} {'매수탈출':<8} {'과매수':<7} {'매도탈출':<8} {'4H수익':<10} {'1D수익':<10} {'평균':<10} {'4H보유':<7} {'1D보유':<7}")
    print("-" * 100)
    
    top = results_df.nlargest(20, 'avg_return')
    
    for i, (_, r) in enumerate(top.iterrows(), 1):
        pos_4h_warn = "⚠️" if r['pos_4h'] > 5 else ""
        pos_1d_warn = "⚠️" if r['pos_1d'] > 3 else ""
        print(f"{i:<3} {int(r['oversold']):<7} {int(r['buy_exit']):<8} {int(r['overbought']):<7} {int(r['sell_exit']):<8} "
              f"{r['return_4h']:+.0f}%{'':<5} {r['return_1d']:+.0f}%{'':<5} {r['avg_return']:+.0f}%{'':<5} "
              f"{int(r['pos_4h'])}{pos_4h_warn:<4} {int(r['pos_1d'])}{pos_1d_warn}")
    
    print()
    print("=" * 100)
    print("🎯 현재 물타기 적은 것 중 평균 수익률 Top 10")
    print("=" * 100)
    print()
    
    # 4시간봉 물타기 5회 이하, 일봉 물타기 2회 이하
    safe = results_df[(results_df['pos_4h'] <= 5) & (results_df['pos_1d'] <= 2)]
    
    if len(safe) > 0:
        top_safe = safe.nlargest(10, 'avg_return')
        for i, (_, r) in enumerate(top_safe.iterrows(), 1):
            print(f"{i:<3} {int(r['oversold']):<7} {int(r['buy_exit']):<8} {int(r['overbought']):<7} {int(r['sell_exit']):<8} "
                  f"{r['return_4h']:+.0f}%{'':<5} {r['return_1d']:+.0f}%{'':<5} {r['avg_return']:+.0f}%{'':<5} "
                  f"{int(r['pos_4h'])}회{'':<3} {int(r['pos_1d'])}회")
    else:
        print("   조건에 맞는 조합 없음")
    
    print()
    print("=" * 100)
    print("🎯 추천 설정")
    print("=" * 100)
    
    best = top.iloc[0]
    print(f"""
   📈 매수 조건
   ─────────────────────────
   과매도 기준: RSI < {int(best['oversold'])}
   매수 탈출:   RSI ≥ {int(best['buy_exit'])}
   
   📉 매도 조건
   ─────────────────────────
   과매수 기준: RSI > {int(best['overbought'])}
   매도 탈출:   RSI ≤ {int(best['sell_exit'])}
   
   📊 성과
   ─────────────────────────
   4시간봉 2년: {best['return_4h']:+.0f}% (현재 {int(best['pos_4h'])}회 보유)
   일봉 5년:    {best['return_1d']:+.0f}% (현재 {int(best['pos_1d'])}회 보유)
   평균 수익률: {best['avg_return']:+.0f}%
""")


if __name__ == '__main__':
    main()
