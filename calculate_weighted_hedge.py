"""
실제 금액 기준 (Capital-Weighted) 헷징 전략 수익률 계산
물타기 횟수 × 투자금 고려
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
CAPITAL_PER_ENTRY = 1000  # 각 진입당 $1000

# 데이터 로드
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()

# 지표 계산
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()
exp12 = df['Close'].ewm(span=12).mean()
exp26 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp12 - exp26

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df['high_60'] = df['High'].rolling(60).max()
df['drawdown_60'] = (df['Close'] - df['high_60']) / df['high_60'] * 100

# 롱 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

print("=" * 120)
print("📊 실제 금액 기준 (Capital-Weighted) 헷징 전략 분석")
print("=" * 120)
print(f"각 진입당 투자금: ${CAPITAL_PER_ENTRY:,}")
print(f"총 거래: {len(trades)}건\n")

# ===== 롱만 했을 때 (기준) =====
def calculate_weighted_return_long_only(trades):
    """롱만 했을 때 금액 기준 수익률"""
    total_invested = 0
    total_profit = 0
    
    for trade in trades:
        invested = trade['num_buys'] * CAPITAL_PER_ENTRY
        profit = invested * (trade['return'] / 100)
        total_invested += invested
        total_profit += profit
    
    weighted_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    return total_invested, total_profit, weighted_return

total_inv_long, total_profit_long, weighted_long = calculate_weighted_return_long_only(trades)

print("🟢 롱만 했을 때 (기준):")
print(f"   총 투자금: ${total_inv_long:,.0f}")
print(f"   총 손익: ${total_profit_long:,.0f}")
print(f"   금액 기준 수익률: {weighted_long:+.2f}%")
print(f"   (단순 수익률 합계: {sum(t['return'] for t in trades):+.1f}%)")

# ===== 헷징 전략 시뮬레이션 =====
def simulate_hedge_weighted(trades, df, avg_threshold, entry_func, profit_target, stop_loss):
    """
    실제 금액 기준 헷징 전략 시뮬레이션
    
    롱 투자금: 물타기 횟수 × $1000
    숏 투자금: 헷징 시점부터 남은 물타기 가정 (간단히 $1000 고정)
    """
    total_long_invested = 0
    total_long_profit = 0
    total_short_invested = 0
    total_short_profit = 0
    hedge_count = 0
    
    for trade in trades:
        entry_dates = trade['entry_dates']
        num_buys = trade['num_buys']
        long_return = trade['return']
        
        # 롱 투자금/손익
        long_invested = num_buys * CAPITAL_PER_ENTRY
        long_profit = long_invested * (long_return / 100)
        total_long_invested += long_invested
        total_long_profit += long_profit
        
        # 헷징 조건 확인
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
        
        # 숏 헷징 실행
        short_entry_price = df['Close'].iloc[idx]
        long_exit_idx = df.index.get_indexer([trade['exit_date']], method='ffill')[0]
        
        target_price = short_entry_price * (1 - profit_target / 100)
        stop_price = short_entry_price * (1 + abs(stop_loss) / 100) if stop_loss < 0 else None
        
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
        
        # 숏 투자금: 헷징 시점 기준으로 롱과 비슷한 규모로 가정
        # (실제로는 헷징 비율을 어떻게 정할지에 따라 달라짐)
        # 여기서는 간단히 현재 롱 투자금의 50%로 가정
        short_invested = long_invested * 0.5
        short_profit = short_invested * (short_return / 100)
        
        total_short_invested += short_invested
        total_short_profit += short_profit
        hedge_count += 1
    
    total_invested = total_long_invested + total_short_invested
    total_profit = total_long_profit + total_short_profit
    
    weighted_return = (total_profit / total_long_invested * 100) if total_long_invested > 0 else 0
    
    return {
        'long_invested': total_long_invested,
        'long_profit': total_long_profit,
        'short_invested': total_short_invested,
        'short_profit': total_short_profit,
        'total_profit': total_profit,
        'weighted_return': weighted_return,
        'hedge_count': hedge_count
    }

# 테스트할 조합들
test_cases = [
    ("물타기2회 + MACD<0 + 수익5%/손절-20%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -20),
    ("물타기2회 + MACD<0 + 수익5%/손절-15%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
    ("물타기2회 + MACD<0 + 수익5%/손절-10%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -10),
    ("물타기2회 + 가격<MA20 + 수익6%", 2, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 6, -100),
    ("물타기2회 + 가격<MA20 + 수익5%/손절-20%", 2, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 5, -20),
    ("물타기2회 + 가격<MA20 + 수익5%/손절-15%", 2, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 5, -15),
    ("물타기3회 + MACD<0 + 수익5%/손절-15%", 3, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
    ("물타기3회 + 가격<MA20 + 수익5%/손절-15%", 3, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 5, -15),
    ("물타기4회 + MACD<0 + 수익5%/손절-15%", 4, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
    ("물타기5회 + MACD<0 + 수익5%/손절-15%", 5, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
]

print("\n" + "=" * 120)
print("📊 헷징 전략별 금액 기준 수익률 비교")
print("=" * 120)

print(f"\n{'전략':<45} | {'헷징':>5} | {'롱손익':>12} | {'숏손익':>12} | {'총손익':>12} | {'금액수익률':>10} | {'롱대비':>10}")
print("-" * 120)

results = []
for name, avg_th, entry_func, profit, stop in test_cases:
    result = simulate_hedge_weighted(trades, df, avg_th, entry_func, profit, stop)
    diff = result['weighted_return'] - weighted_long
    results.append((name, result, diff))
    
    print(f"{name:<45} | {result['hedge_count']:>4}건 | ${result['long_profit']:>+10,.0f} | ${result['short_profit']:>+10,.0f} | ${result['total_profit']:>+10,.0f} | {result['weighted_return']:>+9.2f}% | {diff:>+9.2f}%")

print(f"\n{'롱만 (기준)':<45} | {'-':>5} | ${total_profit_long:>+10,.0f} | ${0:>10} | ${total_profit_long:>+10,.0f} | {weighted_long:>+9.2f}% | {0:>+9.2f}%")

# 최적 조합 찾기
best = max(results, key=lambda x: x[2])
print(f"\n🏆 최적 조합: {best[0]}")
print(f"   금액 기준 수익률: {best[1]['weighted_return']:+.2f}%")
print(f"   롱대비 추가 수익: {best[2]:+.2f}%")

# ===== 더 다양한 조합 테스트 =====
print("\n" + "=" * 120)
print("📊 다양한 물타기 기준별 최적 조합 (금액 기준)")
print("=" * 120)

for avg_threshold in [2, 3, 4, 5, 6]:
    best_for_threshold = None
    best_diff = -999
    
    for entry_name, entry_func in [
        ("MACD<0", lambda df, idx: df['MACD'].iloc[idx] < 0),
        ("가격<MA20", lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx]),
        ("가격<MA50", lambda df, idx: df['Close'].iloc[idx] < df['MA50'].iloc[idx]),
        ("RSI<45", lambda df, idx: df['RSI'].iloc[idx] < 45),
        ("고점-10%", lambda df, idx: df['drawdown_60'].iloc[idx] <= -10),
    ]:
        for profit, stop in [(5, -15), (5, -20), (6, -100), (7, -15)]:
            result = simulate_hedge_weighted(trades, df, avg_threshold, entry_func, profit, stop)
            diff = result['weighted_return'] - weighted_long
            
            if diff > best_diff and result['hedge_count'] >= 3:
                best_diff = diff
                best_for_threshold = (entry_name, profit, stop, result, diff)
    
    if best_for_threshold:
        entry_name, profit, stop, result, diff = best_for_threshold
        stop_str = f"/손절{stop}%" if stop > -100 else ""
        print(f"\n🔹 물타기 {avg_threshold}회 이상:")
        print(f"   최적: {entry_name} + 수익{profit}%{stop_str}")
        print(f"   헷징 발동: {result['hedge_count']}건")
        print(f"   금액 수익률: {result['weighted_return']:+.2f}% (롱대비 {diff:+.2f}%)")

print("\n" + "=" * 120)
print("✅ 분석 완료!")
print("=" * 120)

