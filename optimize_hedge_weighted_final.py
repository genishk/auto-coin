"""
실제 금액 기준 (Capital-Weighted) 헷징 전략 최종 최적화
물타기 2~10회, 다양한 지표, 다양한 청산 조건
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from dashboard_4h import (
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

# 파라미터
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False
CAPITAL_PER_ENTRY = 1000

# 데이터 로드
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()

# 지표 계산
for period in [10, 20, 30, 50, 100]:
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

df['high_60'] = df['High'].rolling(60).max()
df['drawdown_60'] = (df['Close'] - df['high_60']) / df['high_60'] * 100

df['RSI_MA'] = df['RSI'].rolling(14).mean()

# 롱 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

print("=" * 130)
print("📊 실제 금액 기준 헷징 전략 최종 최적화")
print("=" * 130)

# 롱만 기준
def calculate_long_only():
    total_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in trades)
    total_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in trades)
    return total_invested, total_profit, total_profit / total_invested * 100

total_inv, total_profit_long, weighted_long = calculate_long_only()
print(f"롱만: 총 투자 ${total_inv:,} → 손익 ${total_profit_long:,.0f} ({weighted_long:+.2f}%)\n")

# ===== 조합 정의 =====
avg_thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10]

entry_conditions = {
    "무조건": lambda df, idx: True,
    "MACD<0": lambda df, idx: df['MACD'].iloc[idx] < 0,
    "MACD히스토<0": lambda df, idx: df['MACD_hist'].iloc[idx] < 0,
    "MACD<시그널": lambda df, idx: df['MACD'].iloc[idx] < df['MACD_signal'].iloc[idx],
    "가격<MA10": lambda df, idx: df['Close'].iloc[idx] < df['MA10'].iloc[idx],
    "가격<MA20": lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx],
    "가격<MA50": lambda df, idx: df['Close'].iloc[idx] < df['MA50'].iloc[idx],
    "RSI<40": lambda df, idx: df['RSI'].iloc[idx] < 40,
    "RSI<45": lambda df, idx: df['RSI'].iloc[idx] < 45,
    "RSI<50": lambda df, idx: df['RSI'].iloc[idx] < 50,
    "RSI하락추세": lambda df, idx: df['RSI'].iloc[idx] < df['RSI_MA'].iloc[idx],
    "고점-5%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -5,
    "고점-8%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -8,
    "고점-10%": lambda df, idx: df['drawdown_60'].iloc[idx] <= -10,
    "MACD<0+가격<MA20": lambda df, idx: df['MACD'].iloc[idx] < 0 and df['Close'].iloc[idx] < df['MA20'].iloc[idx],
    "MACD<0+RSI<50": lambda df, idx: df['MACD'].iloc[idx] < 0 and df['RSI'].iloc[idx] < 50,
    "가격<MA20+RSI<50": lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx] and df['RSI'].iloc[idx] < 50,
    "고점-8%+MACD<0": lambda df, idx: df['drawdown_60'].iloc[idx] <= -8 and df['MACD'].iloc[idx] < 0,
}

exit_conditions = [
    ("수익3%", 3, None),
    ("수익4%", 4, None),
    ("수익5%", 5, None),
    ("수익6%", 6, None),
    ("수익7%", 7, None),
    ("수익8%", 8, None),
    ("수익10%", 10, None),
    ("수익5%/손절-10%", 5, -10),
    ("수익5%/손절-15%", 5, -15),
    ("수익5%/손절-20%", 5, -20),
    ("수익7%/손절-10%", 7, -10),
    ("수익7%/손절-15%", 7, -15),
    ("수익7%/손절-20%", 7, -20),
    ("수익10%/손절-15%", 10, -15),
]

# ===== 시뮬레이션 =====
def simulate_hedge_weighted(trades, df, avg_threshold, entry_func, profit_target, stop_loss):
    total_long_invested = 0
    total_long_profit = 0
    total_short_invested = 0
    total_short_profit = 0
    hedge_count = 0
    
    for trade in trades:
        entry_dates = trade['entry_dates']
        num_buys = trade['num_buys']
        long_return = trade['return']
        
        long_invested = num_buys * CAPITAL_PER_ENTRY
        long_profit = long_invested * (long_return / 100)
        total_long_invested += long_invested
        total_long_profit += long_profit
        
        if len(entry_dates) < avg_threshold:
            continue
        
        hedge_date = entry_dates[avg_threshold - 1]
        
        try:
            idx = df.index.get_loc(hedge_date)
        except:
            idx = df.index.get_indexer([hedge_date], method='ffill')[0]
        
        if idx < 0 or idx >= len(df):
            continue
        
        try:
            if not entry_func(df, idx):
                continue
        except:
            continue
        
        short_entry_price = df['Close'].iloc[idx]
        long_exit_idx = df.index.get_indexer([trade['exit_date']], method='ffill')[0]
        
        target_price = short_entry_price * (1 - profit_target / 100)
        stop_price = short_entry_price * (1 + abs(stop_loss) / 100) if stop_loss else None
        
        short_exit_price = None
        for i in range(idx + 1, min(long_exit_idx + 1, len(df))):
            if df['Low'].iloc[i] <= target_price:
                short_exit_price = target_price
                break
            if stop_price and df['High'].iloc[i] >= stop_price:
                short_exit_price = stop_price
                break
        
        if short_exit_price is None and long_exit_idx > idx and long_exit_idx < len(df):
            short_exit_price = df['Close'].iloc[long_exit_idx]
        
        if short_exit_price is None:
            continue
        
        short_return = (short_entry_price - short_exit_price) / short_entry_price * 100
        
        # 숏 투자금: 롱 투자금의 50%
        short_invested = long_invested * 0.5
        short_profit = short_invested * (short_return / 100)
        
        total_short_invested += short_invested
        total_short_profit += short_profit
        hedge_count += 1
    
    total_profit = total_long_profit + total_short_profit
    weighted_return = (total_profit / total_long_invested * 100) if total_long_invested > 0 else 0
    
    return {
        'long_profit': total_long_profit,
        'short_profit': total_short_profit,
        'total_profit': total_profit,
        'weighted_return': weighted_return,
        'hedge_count': hedge_count
    }

# ===== 전체 테스트 =====
print("📊 전체 조합 테스트 중...")

results = []
for avg_th in avg_thresholds:
    for entry_name, entry_func in entry_conditions.items():
        for exit_name, profit, stop in exit_conditions:
            result = simulate_hedge_weighted(trades, df, avg_th, entry_func, profit, stop)
            if result['hedge_count'] >= 3:
                diff = result['weighted_return'] - weighted_long
                results.append({
                    'avg_threshold': avg_th,
                    'entry': entry_name,
                    'exit': exit_name,
                    'hedge_count': result['hedge_count'],
                    'short_profit': result['short_profit'],
                    'total_profit': result['total_profit'],
                    'weighted_return': result['weighted_return'],
                    'diff': diff
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('diff', ascending=False)

print(f"총 {len(results_df)}개 유효 조합\n")

# ===== 상위 30개 =====
print("=" * 130)
print("🏆 금액 기준 수익률 상위 30개")
print("=" * 130)

print(f"\n{'물타기':>4} | {'진입조건':<22} | {'청산조건':<16} | {'헷징':>5} | {'숏손익':>12} | {'총손익':>12} | {'금액수익률':>10} | {'롱대비':>10}")
print("-" * 130)

for _, row in results_df.head(30).iterrows():
    print(f"{row['avg_threshold']:>3}회 | {row['entry']:<22} | {row['exit']:<16} | {row['hedge_count']:>4}건 | ${row['short_profit']:>+10,.0f} | ${row['total_profit']:>+10,.0f} | {row['weighted_return']:>+9.2f}% | {row['diff']:>+9.2f}%")

# ===== 물타기별 최적 =====
print("\n" + "=" * 130)
print("📊 물타기 횟수별 최적 조합")
print("=" * 130)

for avg_th in avg_thresholds:
    subset = results_df[results_df['avg_threshold'] == avg_th]
    if len(subset) > 0:
        best = subset.iloc[0]
        print(f"\n🔹 물타기 {avg_th}회 이상:")
        print(f"   {best['entry']} + {best['exit']}")
        print(f"   헷징: {best['hedge_count']}건 | 숏손익: ${best['short_profit']:+,.0f}")
        print(f"   금액수익률: {best['weighted_return']:+.2f}% (롱대비 {best['diff']:+.2f}%)")

# ===== 최종 추천 =====
print("\n" + "=" * 130)
print("🏆 최종 추천")
print("=" * 130)

best_overall = results_df.iloc[0]
print(f"\n최적 조합: 물타기 {best_overall['avg_threshold']}회 + {best_overall['entry']} + {best_overall['exit']}")
print(f"   헷징 발동: {best_overall['hedge_count']}건")
print(f"   숏 손익: ${best_overall['short_profit']:+,.0f}")
print(f"   총 손익: ${best_overall['total_profit']:+,.0f}")
print(f"   금액 수익률: {best_overall['weighted_return']:+.2f}%")
print(f"   롱대비 추가: {best_overall['diff']:+.2f}%")

# 헷징 10건 이상 중 최적
min_10 = results_df[results_df['hedge_count'] >= 10].iloc[0] if len(results_df[results_df['hedge_count'] >= 10]) > 0 else None
if min_10 is not None:
    print(f"\n헷징 10건 이상 중 최적: 물타기 {min_10['avg_threshold']}회 + {min_10['entry']} + {min_10['exit']}")
    print(f"   헷징: {min_10['hedge_count']}건 | 금액수익률: {min_10['weighted_return']:+.2f}% (롱대비 {min_10['diff']:+.2f}%)")

# 헷징 20건 이상 중 최적
min_20 = results_df[results_df['hedge_count'] >= 20].iloc[0] if len(results_df[results_df['hedge_count'] >= 20]) > 0 else None
if min_20 is not None:
    print(f"\n헷징 20건 이상 중 최적: 물타기 {min_20['avg_threshold']}회 + {min_20['entry']} + {min_20['exit']}")
    print(f"   헷징: {min_20['hedge_count']}건 | 금액수익률: {min_20['weighted_return']:+.2f}% (롱대비 {min_20['diff']:+.2f}%)")

print("\n" + "=" * 130)
print("✅ 분석 완료!")
print("=" * 130)

