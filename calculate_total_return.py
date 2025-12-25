"""
전체 수익률 정확히 계산
헷징 안 한 거래 + 헷징한 거래 전부 합산
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
print("📊 전체 수익률 정확한 계산")
print("=" * 100)

# ===== 롱만 했을 때 =====
total_long_return = sum(t['return'] for t in trades)
print(f"\n🟢 롱만 했을 때:")
print(f"   총 거래: {len(trades)}건")
print(f"   총 수익률: {total_long_return:+.1f}%")

# ===== 헷징 전략 적용 =====
def simulate_hedge_strategy(trades, df, avg_threshold, entry_func, profit_target, stop_loss):
    """
    전체 거래에 헷징 전략 적용
    - 조건 충족 시: 롱 + 숏
    - 조건 미충족 시: 롱만
    """
    total_return = 0
    hedged_count = 0
    not_hedged_count = 0
    
    for trade in trades:
        entry_dates = trade['entry_dates']
        long_return = trade['return']
        
        # 물타기 횟수 부족하면 롱만
        if len(entry_dates) < avg_threshold:
            total_return += long_return
            not_hedged_count += 1
            continue
        
        # 헷지 시점
        hedge_date = entry_dates[avg_threshold - 1]
        
        try:
            idx = df.index.get_loc(hedge_date)
        except:
            idx = df.index.get_indexer([hedge_date], method='ffill')[0]
        
        if idx < 0 or idx >= len(df):
            total_return += long_return
            not_hedged_count += 1
            continue
        
        # 진입 조건 확인
        try:
            if not entry_func(df, idx):
                total_return += long_return
                not_hedged_count += 1
                continue
        except:
            total_return += long_return
            not_hedged_count += 1
            continue
        
        # 숏 헷징 실행
        short_entry_price = df['Close'].iloc[idx]
        long_exit_idx = df.index.get_indexer([trade['exit_date']], method='ffill')[0]
        
        target_price = short_entry_price * (1 - profit_target / 100)
        stop_price = short_entry_price * (1 - stop_loss / 100)
        
        short_exit_price = None
        for i in range(idx + 1, min(long_exit_idx + 1, len(df))):
            if df['Low'].iloc[i] <= target_price:
                short_exit_price = target_price
                break
            if df['High'].iloc[i] >= stop_price:
                short_exit_price = stop_price
                break
        
        if short_exit_price is None and long_exit_idx > idx and long_exit_idx < len(df):
            short_exit_price = df['Close'].iloc[long_exit_idx]
        
        if short_exit_price is None:
            total_return += long_return
            not_hedged_count += 1
            continue
        
        short_return = (short_entry_price - short_exit_price) / short_entry_price * 100
        total_return += long_return + short_return
        hedged_count += 1
    
    return total_return, hedged_count, not_hedged_count

# 테스트할 조합들
test_cases = [
    ("물타기2회 + MACD<0 + 수익5%/손절-20%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -20),
    ("물타기2회 + MACD<0 + 수익5%/손절-15%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
    ("물타기2회 + MACD<0 + 수익5%/손절-10%", 2, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -10),
    ("물타기2회 + 가격<MA20 + 수익6%", 2, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 6, -100),
    ("물타기2회 + 가격<MA20 + 수익5%/손절-20%", 2, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 5, -20),
    ("물타기3회 + MACD<0 + 수익5%/손절-15%", 3, lambda df, idx: df['MACD'].iloc[idx] < 0, 5, -15),
    ("물타기3회 + 가격<MA20 + 수익5%/손절-15%", 3, lambda df, idx: df['Close'].iloc[idx] < df['MA20'].iloc[idx], 5, -15),
]

print("\n" + "=" * 100)
print("📊 헷징 전략별 전체 수익률 비교")
print("=" * 100)

print(f"\n{'전략':<45} | {'헷징발동':>8} | {'헷징안함':>8} | {'전체수익률':>12} | {'롱대비':>10}")
print("-" * 100)

for name, avg_th, entry_func, profit, stop in test_cases:
    total, hedged, not_hedged = simulate_hedge_strategy(trades, df, avg_th, entry_func, profit, stop)
    diff = total - total_long_return
    print(f"{name:<45} | {hedged:>7}건 | {not_hedged:>7}건 | {total:>+11.1f}% | {diff:>+9.1f}%")

print(f"\n{'롱만 (기준)':<45} | {'-':>8} | {len(trades):>7}건 | {total_long_return:>+11.1f}% | {0:>+9.1f}%")

print("\n" + "=" * 100)
print("✅ 분석 완료!")
print("=" * 100)

