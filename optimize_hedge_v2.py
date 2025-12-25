"""
숏 헷징 전략 최적화 v2
다양한 조합 테스트 - 총 헷징 수익 기준
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from itertools import product

from dashboard_4h import (
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

print("=" * 120)
print("📊 숏 헷징 전략 최적화 v2 - 총 헷징 수익 기준")
print("=" * 120)

# 파라미터
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False

# 데이터 로드
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()
print(f"데이터: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)\n")

# 지표 계산
print("📈 지표 계산 중...")

for period in [20, 50, 100, 200]:
    df[f'MA{period}'] = df['Close'].rolling(period).mean()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

exp12 = df['Close'].ewm(span=12).mean()
exp26 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp12 - exp26
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# 다양한 lookback으로 고점대비 하락률
for lookback in [30, 45, 60, 90, 120]:
    df[f'high_{lookback}'] = df['High'].rolling(lookback).max()
    df[f'drawdown_{lookback}'] = (df['Close'] - df[f'high_{lookback}']) / df[f'high_{lookback}'] * 100

df['RSI_MA'] = df['RSI'].rolling(14).mean()

# 롱 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

print(f"총 롱 거래: {len(trades)}회\n")

# ===== 다양한 조합 정의 =====

# 1. 물타기 횟수
avg_thresholds = [2, 3, 4, 5]

# 2. 진입 조건 (더 다양하게)
entry_conditions = {
    # 무조건
    "무조건": lambda df, idx: True,
    
    # MACD 기반
    "MACD<0": lambda df, idx: df['MACD'].iloc[idx] < 0,
    "MACD히스토<0": lambda df, idx: df['MACD_hist'].iloc[idx] < 0,
    
    # 가격 vs MA
    "가격<MA20": lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx],
    "가격<MA50": lambda df, idx: df['Close'].iloc[idx] < df['MA50'].iloc[idx],
    "가격<MA100": lambda df, idx: df['Close'].iloc[idx] < df['MA100'].iloc[idx],
    
    # 고점대비 (다양한 기준)
    "고점-8%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -8,
    "고점-10%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -10,
    "고점-12%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -12,
    "고점-15%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -15,
    "고점-18%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -18,
    "고점-20%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -20,
    
    # RSI 기반
    "RSI<45": lambda df, idx: df['RSI'].iloc[idx] < 45,
    "RSI<50": lambda df, idx: df['RSI'].iloc[idx] < 50,
    "RSI하락추세": lambda df, idx: df['RSI'].iloc[idx] < df['RSI_MA'].iloc[idx],
    
    # 복합 조건
    "MACD<0+가격<MA50": lambda df, idx: df['MACD'].iloc[idx] < 0 and df['Close'].iloc[idx] < df['MA50'].iloc[idx],
    "MACD<0+가격<MA100": lambda df, idx: df['MACD'].iloc[idx] < 0 and df['Close'].iloc[idx] < df['MA100'].iloc[idx],
    "고점-8%+MACD<0": lambda df, idx: df['drawdown_60'].iloc[idx] <= -8 and df['MACD'].iloc[idx] < 0,
    "고점-10%+MACD<0": lambda df, idx: df['drawdown_60'].iloc[idx] <= -10 and df['MACD'].iloc[idx] < 0,
    "고점-12%+MACD<0": lambda df, idx: df['drawdown_60'].iloc[idx] <= -12 and df['MACD'].iloc[idx] < 0,
    "고점-15%+MACD<0": lambda df, idx: df['drawdown_60'].iloc[idx] <= -15 and df['MACD'].iloc[idx] < 0,
    "고점-10%+RSI<50": lambda df, idx: df['drawdown_60'].iloc[idx] <= -10 and df['RSI'].iloc[idx] < 50,
    "고점-15%+RSI<50": lambda df, idx: df['drawdown_60'].iloc[idx] <= -15 and df['RSI'].iloc[idx] < 50,
    "가격<MA50+RSI<50": lambda df, idx: df['Close'].iloc[idx] < df['MA50'].iloc[idx] and df['RSI'].iloc[idx] < 50,
    "MACD<0+RSI<50": lambda df, idx: df['MACD'].iloc[idx] < 0 and df['RSI'].iloc[idx] < 50,
    "MACD히스토<0+RSI<50": lambda df, idx: df['MACD_hist'].iloc[idx] < 0 and df['RSI'].iloc[idx] < 50,
    
    # 3중 조건
    "고점-10%+MACD<0+RSI<50": lambda df, idx: df['drawdown_60'].iloc[idx] <= -10 and df['MACD'].iloc[idx] < 0 and df['RSI'].iloc[idx] < 50,
    "가격<MA50+MACD<0+RSI<50": lambda df, idx: df['Close'].iloc[idx] < df['MA50'].iloc[idx] and df['MACD'].iloc[idx] < 0 and df['RSI'].iloc[idx] < 50,
}

# 3. 청산 조건 (더 다양하게)
exit_conditions = {
    # 롱과 함께
    "롱청산시": ("with_long", None),
    
    # 수익 목표
    "수익2%": ("profit", 2),
    "수익3%": ("profit", 3),
    "수익4%": ("profit", 4),
    "수익5%": ("profit", 5),
    "수익6%": ("profit", 6),
    "수익7%": ("profit", 7),
    "수익8%": ("profit", 8),
    "수익10%": ("profit", 10),
    
    # 손절
    "손절-5%": ("stop", -5),
    "손절-8%": ("stop", -8),
    "손절-10%": ("stop", -10),
    "손절-15%": ("stop", -15),
    
    # 기간
    "최대20봉": ("bars", 20),
    "최대30봉": ("bars", 30),
    "최대45봉": ("bars", 45),
    "최대60봉": ("bars", 60),
    
    # 복합: 수익 목표 + 손절
    "수익3%/손절-10%": ("profit_stop", 3, -10),
    "수익5%/손절-10%": ("profit_stop", 5, -10),
    "수익5%/손절-15%": ("profit_stop", 5, -15),
    "수익7%/손절-10%": ("profit_stop", 7, -10),
}

# ===== 시뮬레이션 함수 =====
def simulate_hedge(trade, df, avg_threshold, entry_func, exit_type, exit_param1=None, exit_param2=None):
    entry_dates = trade['entry_dates']
    
    if len(entry_dates) < avg_threshold:
        return None
    
    hedge_date = entry_dates[avg_threshold - 1]
    
    try:
        idx = df.index.get_loc(hedge_date)
    except:
        idx = df.index.get_indexer([hedge_date], method='ffill')[0]
    
    if idx < 0 or idx >= len(df):
        return None
    
    try:
        if not entry_func(df, idx):
            return None
    except:
        return None
    
    short_entry_price = df['Close'].iloc[idx]
    short_entry_date = df.index[idx]
    
    long_exit_idx = df.index.get_indexer([trade['exit_date']], method='ffill')[0]
    
    short_exit_price = None
    short_exit_date = None
    
    if exit_type == "with_long":
        if long_exit_idx > idx and long_exit_idx < len(df):
            short_exit_price = df['Close'].iloc[long_exit_idx]
            short_exit_date = df.index[long_exit_idx]
    
    elif exit_type == "profit":
        target_price = short_entry_price * (1 - exit_param1 / 100)
        for i in range(idx + 1, min(long_exit_idx + 1, len(df))):
            if df['Low'].iloc[i] <= target_price:
                short_exit_price = target_price
                short_exit_date = df.index[i]
                break
        if short_exit_price is None and long_exit_idx > idx:
            short_exit_price = df['Close'].iloc[long_exit_idx]
            short_exit_date = df.index[long_exit_idx]
    
    elif exit_type == "stop":
        stop_price = short_entry_price * (1 - exit_param1 / 100)  # exit_param1 is negative
        for i in range(idx + 1, min(long_exit_idx + 1, len(df))):
            if df['High'].iloc[i] >= stop_price:
                short_exit_price = stop_price
                short_exit_date = df.index[i]
                break
        if short_exit_price is None and long_exit_idx > idx:
            short_exit_price = df['Close'].iloc[long_exit_idx]
            short_exit_date = df.index[long_exit_idx]
    
    elif exit_type == "bars":
        exit_idx = min(idx + exit_param1, long_exit_idx, len(df) - 1)
        short_exit_price = df['Close'].iloc[exit_idx]
        short_exit_date = df.index[exit_idx]
    
    elif exit_type == "profit_stop":
        target_price = short_entry_price * (1 - exit_param1 / 100)
        stop_price = short_entry_price * (1 - exit_param2 / 100)
        for i in range(idx + 1, min(long_exit_idx + 1, len(df))):
            if df['Low'].iloc[i] <= target_price:
                short_exit_price = target_price
                short_exit_date = df.index[i]
                break
            if df['High'].iloc[i] >= stop_price:
                short_exit_price = stop_price
                short_exit_date = df.index[i]
                break
        if short_exit_price is None and long_exit_idx > idx:
            short_exit_price = df['Close'].iloc[long_exit_idx]
            short_exit_date = df.index[long_exit_idx]
    
    if short_exit_price is None:
        return None
    
    short_return = (short_entry_price - short_exit_price) / short_entry_price * 100
    
    return {
        'short_return': short_return,
        'long_return': trade['return'],
        'combined_return': short_return + trade['return'],
        'long_num_buys': trade['num_buys']
    }

# ===== 전체 조합 테스트 =====
print("=" * 120)
print("📊 조합별 성과 테스트 (다양한 조합)")
print("=" * 120)

results = []
total_combos = len(avg_thresholds) * len(entry_conditions) * len(exit_conditions)
print(f"총 {total_combos}개 조합 테스트 중...\n")

for avg_threshold in avg_thresholds:
    for entry_name, entry_func in entry_conditions.items():
        for exit_name, exit_params in exit_conditions.items():
            
            if isinstance(exit_params, tuple):
                if len(exit_params) == 2:
                    exit_type, exit_param1 = exit_params
                    exit_param2 = None
                else:
                    exit_type, exit_param1, exit_param2 = exit_params
            else:
                exit_type = exit_params
                exit_param1 = exit_param2 = None
            
            hedge_results = []
            for trade in trades:
                result = simulate_hedge(trade, df, avg_threshold, entry_func, exit_type, exit_param1, exit_param2)
                if result:
                    hedge_results.append(result)
            
            if len(hedge_results) >= 2:
                total_short_return = sum(r['short_return'] for r in hedge_results)
                total_long_return = sum(r['long_return'] for r in hedge_results)
                total_combined = sum(r['combined_return'] for r in hedge_results)
                win_rate = len([r for r in hedge_results if r['short_return'] > 0]) / len(hedge_results) * 100
                avg_short = total_short_return / len(hedge_results)
                
                results.append({
                    'avg_threshold': avg_threshold,
                    'entry_condition': entry_name,
                    'exit_condition': exit_name,
                    'count': len(hedge_results),
                    'win_rate': win_rate,
                    'total_short_return': total_short_return,
                    'total_long_return': total_long_return,
                    'total_combined': total_combined,
                    'hedge_effect': total_short_return,
                    'avg_short': avg_short
                })

# 총 헷징 수익 기준 정렬
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_short_return', ascending=False)

print(f"총 {len(results_df)}개 유효 조합\n")

# ===== 상위 결과 =====
print("=" * 120)
print("🏆 총 헷징 수익 상위 30개")
print("=" * 120)

print(f"\n{'물타기':>4} | {'진입조건':<25} | {'청산조건':<18} | {'횟수':>4} | {'승률':>6} | {'숏총수익':>10} | {'숏평균':>8} | {'롱총':>10} | {'합계총':>10}")
print("-" * 130)

for _, row in results_df.head(30).iterrows():
    print(f"{row['avg_threshold']:>3}회 | {row['entry_condition']:<25} | {row['exit_condition']:<18} | {row['count']:>4} | {row['win_rate']:>5.1f}% | {row['total_short_return']:>+9.1f}% | {row['avg_short']:>+7.2f}% | {row['total_long_return']:>+9.1f}% | {row['total_combined']:>+9.1f}%")

# ===== 물타기 횟수별 최적 =====
print("\n" + "=" * 120)
print("📊 물타기 횟수별 최적 조합 (총 헷징 수익 기준)")
print("=" * 120)

for threshold in avg_thresholds:
    subset = results_df[results_df['avg_threshold'] == threshold]
    if len(subset) > 0:
        best = subset.iloc[0]
        print(f"\n🔹 물타기 {threshold}회 이상:")
        print(f"   진입: {best['entry_condition']} | 청산: {best['exit_condition']}")
        print(f"   발생: {best['count']}회 | 승률: {best['win_rate']:.1f}%")
        print(f"   총 숏수익: {best['total_short_return']:+.1f}% | 평균: {best['avg_short']:+.2f}%")

# ===== 건수 10건 이상 중 최적 =====
print("\n" + "=" * 120)
print("📊 건수 10건 이상 중 최적 조합")
print("=" * 120)

min_10 = results_df[results_df['count'] >= 10].head(10)
print(f"\n{'물타기':>4} | {'진입조건':<25} | {'청산조건':<18} | {'횟수':>4} | {'승률':>6} | {'숏총수익':>10}")
print("-" * 100)

for _, row in min_10.iterrows():
    print(f"{row['avg_threshold']:>3}회 | {row['entry_condition']:<25} | {row['exit_condition']:<18} | {row['count']:>4} | {row['win_rate']:>5.1f}% | {row['total_short_return']:>+9.1f}%")

# ===== 건수 20건 이상 중 최적 =====
print("\n" + "=" * 120)
print("📊 건수 20건 이상 중 최적 조합")
print("=" * 120)

min_20 = results_df[results_df['count'] >= 20].head(10)
print(f"\n{'물타기':>4} | {'진입조건':<25} | {'청산조건':<18} | {'횟수':>4} | {'승률':>6} | {'숏총수익':>10}")
print("-" * 100)

for _, row in min_20.iterrows():
    print(f"{row['avg_threshold']:>3}회 | {row['entry_condition']:<25} | {row['exit_condition']:<18} | {row['count']:>4} | {row['win_rate']:>5.1f}% | {row['total_short_return']:>+9.1f}%")

print("\n" + "=" * 120)
print("✅ 분석 완료!")
print("=" * 120)

