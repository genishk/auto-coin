"""
새 전략 (수익일 때만 매도) 최적화
- 약 300개 조합 테스트
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
    max_positions = 0  # 최대 물타기 횟수 추적
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df.iloc[idx]['Close']
        
        if positions:
            avg_price = sum(p['price'] for p in positions) / len(positions)
            current_return = (current_price / avg_price - 1) * 100
            
            max_positions = max(max_positions, len(positions))
            
            exit_reason = None
            exit_price = current_price
            
            # 손절은 무조건
            if current_return <= stop_loss:
                exit_reason = "손절"
            # RSI 매도 + 수익일 때만
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({
                    'return': final_return,
                    'num_buys': len(positions)
                })
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    # 현재 보유 중인 포지션 수도 반환
    return trades, len(positions), max_positions


def calculate_metrics(trades, current_positions, max_positions):
    if not trades:
        return {
            'num_trades': 0, 'win_rate': 0, 'avg_return': 0, 
            'total_return': 0, 'current_pos': current_positions,
            'max_positions': max_positions
        }
    
    wins = [t for t in trades if t['return'] > 0]
    returns = [t['return'] for t in trades]
    
    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r / 100)
    total_return = (cumulative - 1) * 100
    
    return {
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'avg_return': np.mean(returns),
        'total_return': total_return,
        'current_pos': current_positions,
        'max_positions': max_positions
    }


def main():
    print("=" * 80)
    print("🔬 새 전략 (수익일 때만 매도) RSI 기준 최적화 - 일봉 5년")
    print("=" * 80)
    print()
    
    # 데이터 로드 (일봉 5년 - 하락장 포함!)
    print("📊 데이터 로딩 중 (일봉 5년 - 2022 하락장 포함)...")
    ticker = 'BTC-USD'
    cache = DataCache(cache_dir='data/cache', max_age_hours=24)
    df = cache.get(f'{ticker}_1d')
    
    if df is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='5y', interval='1d')
        df = data[ticker]
        df, _ = validate_data(df, ticker)
        cache.set(f'{ticker}_1d', df)
    
    ti = TechnicalIndicators()
    df = ti.calculate_all(df)
    
    print(f"   데이터: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print()
    
    # 파라미터 범위 (약 300개 조합)
    oversold_range = [20, 25, 30, 35]              # 4개
    buy_exit_range = [30, 35, 40, 45, 50]          # 5개
    overbought_range = [65, 70, 75, 80]            # 4개
    sell_exit_range = [30, 35, 40, 45]             # 4개
    
    # 4 × 5 × 4 × 4 = 320개
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
        
        buy_signals = find_buy_signals(df, oversold, buy_exit)
        sell_signals = find_sell_signals(df, overbought, sell_exit)
        trades, current_pos, max_pos = simulate_new_strategy(df, buy_signals, sell_signals, stop_loss=-25)
        metrics = calculate_metrics(trades, current_pos, max_pos)
        
        # 현재 물타기 너무 많으면 패널티
        if metrics['num_trades'] >= 3:  # 최소 거래 3회
            results.append({
                'oversold': oversold, 'buy_exit': buy_exit,
                'overbought': overbought, 'sell_exit': sell_exit,
                **metrics
            })
    
    results_df = pd.DataFrame(results)
    
    # 현재 포지션 적은 것 우선, 그 다음 수익률
    results_df['score'] = results_df['total_return'] - results_df['current_pos'] * 10
    
    print()
    print("=" * 95)
    print("🏆 총 수익률 기준 Top 15 (현재 물타기 상태 고려)")
    print("=" * 95)
    print()
    print(f"{'#':<3} {'과매도':<7} {'매수탈출':<8} {'과매수':<7} {'매도탈출':<8} {'거래':<6} {'승률':<8} {'수익률':<10} {'현재보유':<8} {'최대물타기':<10}")
    print("-" * 95)
    
    # 현재 포지션 0인 것 중에서 수익률 높은 것
    zero_pos = results_df[results_df['current_pos'] == 0].nlargest(10, 'total_return')
    
    for i, (_, r) in enumerate(zero_pos.iterrows(), 1):
        print(f"{i:<3} {int(r['oversold']):<7} {int(r['buy_exit']):<8} {int(r['overbought']):<7} {int(r['sell_exit']):<8} "
              f"{int(r['num_trades']):<6} {r['win_rate']:.0f}%{'':<4} {r['total_return']:+.0f}%{'':<5} "
              f"{int(r['current_pos'])}회{'':<4} {int(r['max_positions'])}회")
    
    print()
    print("=" * 95)
    print("📊 수익률 Top 15 (현재 포지션 무관)")
    print("=" * 95)
    print()
    
    top_return = results_df.nlargest(15, 'total_return')
    
    for i, (_, r) in enumerate(top_return.iterrows(), 1):
        pos_warning = "⚠️" if r['current_pos'] > 5 else ""
        print(f"{i:<3} {int(r['oversold']):<7} {int(r['buy_exit']):<8} {int(r['overbought']):<7} {int(r['sell_exit']):<8} "
              f"{int(r['num_trades']):<6} {r['win_rate']:.0f}%{'':<4} {r['total_return']:+.0f}%{'':<5} "
              f"{int(r['current_pos'])}회{pos_warning:<3} {int(r['max_positions'])}회")
    
    print()
    print("=" * 95)
    print("🎯 추천 설정 (현재 포지션 0, 수익률 최고)")
    print("=" * 95)
    
    if len(zero_pos) > 0:
        best = zero_pos.iloc[0]
        print(f"""
   📈 매수 조건
   ─────────────────────────
   과매도 기준: RSI < {int(best['oversold'])}
   매수 탈출:   RSI ≥ {int(best['buy_exit'])}
   
   📉 매도 조건
   ─────────────────────────
   과매수 기준: RSI > {int(best['overbought'])}
   매도 탈출:   RSI ≤ {int(best['sell_exit'])}
   
   📊 성과 (2년)
   ─────────────────────────
   거래 횟수: {int(best['num_trades'])}회
   승률: {best['win_rate']:.0f}%
   총 수익률: {best['total_return']:+.0f}%
   현재 보유: {int(best['current_pos'])}회 (청산 완료!)
""")
    else:
        print("\n   ⚠️ 현재 포지션 0인 조합이 없습니다. 위 결과 참고하세요.")


if __name__ == '__main__':
    main()

