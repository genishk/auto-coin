"""
RSI 파라미터 최적화 스크립트
- 실제 매수/매도 시점 (confirm_date/confirm_price) 기준
- Grid Search로 최적 조합 찾기
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


def find_buy_signals(df: pd.DataFrame, rsi_oversold: int, rsi_exit: int) -> list:
    """매수 시그널 찾기 (confirm_date 기준)"""
    signals = []
    in_oversold = False
    signal_start = None
    signal_price = None
    signal_rsi = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi < rsi_oversold and not in_oversold:
            in_oversold = True
            signal_start = df.index[i]
            signal_price = row['Close']
            signal_rsi = current_rsi
        elif in_oversold and current_rsi >= rsi_exit:
            # 실제 매수 시점!
            signals.append({
                'signal_date': signal_start,
                'signal_price': signal_price,
                'signal_rsi': signal_rsi,
                'confirm_date': df.index[i],
                'confirm_price': row['Close'],
                'confirm_rsi': current_rsi
            })
            in_oversold = False
            signal_start = None
    
    return signals


def find_sell_signals(df: pd.DataFrame, rsi_overbought: int, rsi_exit: int) -> list:
    """매도 시그널 찾기 (confirm_date 기준)"""
    signals = []
    in_overbought = False
    signal_start = None
    signal_price = None
    signal_rsi = None
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi > rsi_overbought and not in_overbought:
            in_overbought = True
            signal_start = df.index[i]
            signal_price = row['Close']
            signal_rsi = current_rsi
        elif in_overbought and current_rsi <= rsi_exit:
            # 실제 매도 시점!
            signals.append({
                'signal_date': signal_start,
                'signal_price': signal_price,
                'signal_rsi': signal_rsi,
                'confirm_date': df.index[i],
                'confirm_price': row['Close'],
                'confirm_rsi': current_rsi
            })
            in_overbought = False
            signal_start = None
    
    return signals


def simulate_trades(df: pd.DataFrame, buy_signals: list, sell_signals: list, stop_loss: float = -25):
    """
    물타기 전략 시뮬레이션 (confirm_date/confirm_price 기준)
    """
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
            
            if current_date in all_sell_dates:
                exit_reason = "RSI 매도"
                exit_price = all_sell_dates[current_date]['confirm_price']
            elif current_return <= stop_loss:
                exit_reason = "손절"
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({
                    'entry_date': positions[0]['date'],
                    'exit_date': current_date,
                    'num_buys': len(positions),
                    'avg_price': avg_price,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades


def calculate_metrics(trades: list) -> dict:
    """성과 지표 계산"""
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0
        }
    
    wins = [t for t in trades if t['return'] > 0]
    returns = [t['return'] for t in trades]
    
    # 복리 수익률 계산
    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r / 100)
    total_return = (cumulative - 1) * 100
    
    return {
        'num_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'avg_return': np.mean(returns),
        'total_return': total_return
    }


def main():
    print("=" * 60)
    print("🔍 RSI 파라미터 최적화 (실제 매수/매도 시점 기준)")
    print("=" * 60)
    print()
    
    # 데이터 로드
    print("📊 데이터 로딩 중...")
    ticker = 'BTC-USD'
    cache = DataCache(cache_dir='data/cache_4h', max_age_hours=4)
    df = cache.get(f'{ticker}_4h')
    
    if df is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        df = data[ticker]
        df, _ = validate_data(df, ticker)
        cache.set(f'{ticker}_4h', df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators()
    df = ti.calculate_all(df)
    
    print(f"   데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   데이터 포인트: {len(df):,}개")
    print()
    
    # 파라미터 범위 설정 (간격 10으로 빠르게)
    oversold_range = list(range(20, 45, 10))      # 20, 30, 40
    buy_exit_range = list(range(40, 85, 10))      # 40, 50, 60, 70, 80
    overbought_range = list(range(60, 85, 10))    # 60, 70, 80
    sell_exit_range = list(range(20, 55, 10))     # 20, 30, 40, 50
    
    total_combinations = len(oversold_range) * len(buy_exit_range) * \
                         len(overbought_range) * len(sell_exit_range)
    
    print(f"🔄 총 {total_combinations:,}개 조합 테스트 중...")
    print()
    
    results = []
    
    # Grid Search
    combinations = list(product(
        oversold_range,
        buy_exit_range,
        overbought_range,
        sell_exit_range
    ))
    
    for oversold, buy_exit, overbought, sell_exit in tqdm(combinations, desc="최적화 진행"):
        # 매수 탈출이 과매도보다 커야 함
        if buy_exit <= oversold:
            continue
        # 매도 탈출이 과매수보다 작아야 함
        if sell_exit >= overbought:
            continue
        
        buy_signals = find_buy_signals(df, oversold, buy_exit)
        sell_signals = find_sell_signals(df, overbought, sell_exit)
        trades = simulate_trades(df, buy_signals, sell_signals, stop_loss=-25)
        metrics = calculate_metrics(trades)
        
        results.append({
            'oversold': oversold,
            'buy_exit': buy_exit,
            'overbought': overbought,
            'sell_exit': sell_exit,
            **metrics
        })
    
    # 결과 정렬 (총 수익률 기준)
    results_df = pd.DataFrame(results)
    
    # 최소 거래 횟수 필터 (너무 적으면 신뢰도 낮음)
    results_df = results_df[results_df['num_trades'] >= 5]
    
    # 총 수익률 기준 Top 20
    top_by_return = results_df.nlargest(20, 'total_return')
    
    print()
    print("=" * 80)
    print("🏆 총 수익률 기준 Top 20")
    print("=" * 80)
    print()
    print(f"{'순위':<4} {'과매도':<8} {'매수탈출':<10} {'과매수':<8} {'매도탈출':<10} {'거래수':<8} {'승률':<10} {'평균수익':<12} {'총수익률':<12}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_by_return.iterrows(), 1):
        print(f"{i:<4} {row['oversold']:<8} {row['buy_exit']:<10} {row['overbought']:<8} {row['sell_exit']:<10} "
              f"{row['num_trades']:<8} {row['win_rate']:.1f}%{'':<5} {row['avg_return']:+.1f}%{'':<6} {row['total_return']:+.1f}%")
    
    print()
    print("=" * 80)
    print("📊 승률 기준 Top 10 (거래 10회 이상)")
    print("=" * 80)
    print()
    
    top_by_winrate = results_df[results_df['num_trades'] >= 10].nlargest(10, 'win_rate')
    
    print(f"{'순위':<4} {'과매도':<8} {'매수탈출':<10} {'과매수':<8} {'매도탈출':<10} {'거래수':<8} {'승률':<10} {'평균수익':<12} {'총수익률':<12}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_by_winrate.iterrows(), 1):
        print(f"{i:<4} {row['oversold']:<8} {row['buy_exit']:<10} {row['overbought']:<8} {row['sell_exit']:<10} "
              f"{row['num_trades']:<8} {row['win_rate']:.1f}%{'':<5} {row['avg_return']:+.1f}%{'':<6} {row['total_return']:+.1f}%")
    
    print()
    print("=" * 80)
    print("⚖️ 균형 점수 Top 10 (승률 × 평균수익 × log(거래수))")
    print("=" * 80)
    print()
    
    # 균형 점수 계산
    results_df['balance_score'] = (results_df['win_rate'] / 100) * results_df['avg_return'] * np.log1p(results_df['num_trades'])
    top_balanced = results_df[results_df['num_trades'] >= 5].nlargest(10, 'balance_score')
    
    print(f"{'순위':<4} {'과매도':<8} {'매수탈출':<10} {'과매수':<8} {'매도탈출':<10} {'거래수':<8} {'승률':<10} {'평균수익':<12} {'총수익률':<12}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(top_balanced.iterrows(), 1):
        print(f"{i:<4} {row['oversold']:<8} {row['buy_exit']:<10} {row['overbought']:<8} {row['sell_exit']:<10} "
              f"{row['num_trades']:<8} {row['win_rate']:.1f}%{'':<5} {row['avg_return']:+.1f}%{'':<6} {row['total_return']:+.1f}%")
    
    print()
    print("=" * 80)
    print("🎯 추천 설정")
    print("=" * 80)
    
    # 가장 균형 잡힌 설정 추천
    best = top_balanced.iloc[0]
    print()
    print(f"   과매도 기준: RSI < {int(best['oversold'])}")
    print(f"   매수 탈출:   RSI >= {int(best['buy_exit'])}")
    print(f"   과매수 기준: RSI > {int(best['overbought'])}")
    print(f"   매도 탈출:   RSI <= {int(best['sell_exit'])}")
    print()
    print(f"   예상 거래 횟수: {int(best['num_trades'])}회 (2년 기준)")
    print(f"   예상 승률: {best['win_rate']:.1f}%")
    print(f"   예상 평균 수익률: {best['avg_return']:+.1f}%")
    print(f"   예상 총 수익률: {best['total_return']:+.1f}%")
    print()


if __name__ == '__main__':
    main()

