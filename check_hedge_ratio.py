"""
물타기 2회 시점에서 MACD<0 조건 해당 비율 확인
"""
import sys
sys.path.insert(0, '.')

import pandas as pd

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

# 데이터 로드
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()

# 지표 계산
df['MA20'] = df['Close'].rolling(20).mean()
exp12 = df['Close'].ewm(span=12).mean()
exp26 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp12 - exp26

# 롱 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

print("=" * 100)
print("📊 물타기 2회 시점 조건 해당 비율 분석")
print("=" * 100)

# 물타기 2회 이상인 거래
trades_with_2plus = [t for t in trades if t['num_buys'] >= 2]

print(f"\n전체 거래: {len(trades)}회")
print(f"물타기 2회 이상 거래: {len(trades_with_2plus)}회")

# 물타기 2회 시점에서 조건 확인
macd_negative_count = 0
below_ma20_count = 0
both_count = 0

for trade in trades_with_2plus:
    # 물타기 2회 시점 (= 3번째 구매)
    if len(trade['entry_dates']) >= 2:
        hedge_date = trade['entry_dates'][1]  # 2번째 진입 = 물타기 1회
    else:
        continue
    
    try:
        idx = df.index.get_loc(hedge_date)
    except:
        idx = df.index.get_indexer([hedge_date], method='ffill')[0]
    
    if idx < 0 or idx >= len(df):
        continue
    
    macd = df['MACD'].iloc[idx]
    close = df['Close'].iloc[idx]
    ma20 = df['MA20'].iloc[idx]
    
    if macd < 0:
        macd_negative_count += 1
    if close < ma20:
        below_ma20_count += 1
    if macd < 0 and close < ma20:
        both_count += 1

print(f"\n물타기 2회 시점 (= 2번째 진입)에서 조건 해당:")
print(f"  MACD < 0: {macd_negative_count}/{len(trades_with_2plus)}회 ({macd_negative_count/len(trades_with_2plus)*100:.1f}%)")
print(f"  가격 < MA20: {below_ma20_count}/{len(trades_with_2plus)}회 ({below_ma20_count/len(trades_with_2plus)*100:.1f}%)")
print(f"  둘 다: {both_count}/{len(trades_with_2plus)}회 ({both_count/len(trades_with_2plus)*100:.1f}%)")

# 물타기 횟수별 상세
print("\n" + "=" * 100)
print("📊 물타기 횟수별 분포")
print("=" * 100)

from collections import Counter
buy_counts = Counter([t['num_buys'] for t in trades])

print(f"\n{'물타기':>6} | {'건수':>6} | {'비율':>8}")
print("-" * 30)

for count in sorted(buy_counts.keys()):
    print(f"{count:>5}회 | {buy_counts[count]:>5}회 | {buy_counts[count]/len(trades)*100:>7.1f}%")

# 참고: 물타기 3회 시점도 확인
print("\n" + "=" * 100)
print("📊 물타기 3회 시점 (= 3번째 진입) 조건 해당 비율")
print("=" * 100)

trades_with_3plus = [t for t in trades if t['num_buys'] >= 3]
macd_negative_3 = 0
below_ma20_3 = 0

for trade in trades_with_3plus:
    if len(trade['entry_dates']) >= 3:
        hedge_date = trade['entry_dates'][2]  # 3번째 진입 = 물타기 2회
    else:
        continue
    
    try:
        idx = df.index.get_loc(hedge_date)
    except:
        idx = df.index.get_indexer([hedge_date], method='ffill')[0]
    
    if idx < 0 or idx >= len(df):
        continue
    
    macd = df['MACD'].iloc[idx]
    close = df['Close'].iloc[idx]
    ma20 = df['MA20'].iloc[idx]
    
    if macd < 0:
        macd_negative_3 += 1
    if close < ma20:
        below_ma20_3 += 1

print(f"\n물타기 3회 이상 거래: {len(trades_with_3plus)}회")
print(f"물타기 3회 시점 (= 3번째 진입)에서:")
print(f"  MACD < 0: {macd_negative_3}/{len(trades_with_3plus)}회 ({macd_negative_3/len(trades_with_3plus)*100:.1f}%)")
print(f"  가격 < MA20: {below_ma20_3}/{len(trades_with_3plus)}회 ({below_ma20_3/len(trades_with_3plus)*100:.1f}%)")

print("\n" + "=" * 100)

