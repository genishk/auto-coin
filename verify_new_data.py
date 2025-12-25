"""
새로 수집한 OKX 데이터로 대시보드와 동일한 결과가 나오는지 검증
현재 대시보드: 2년치 (2023-12-24 ~ 2025-12-24)
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta

# 대시보드 함수 import
from dashboard_4h_dual import (
    load_data,
    find_long_signals,
    find_long_exit_signals,
    find_short_signals,
    find_short_exit_signals,
    simulate_dual_trades
)

print("=" * 100)
print("🔍 새 데이터 vs 대시보드 검증")
print("=" * 100)

# 파라미터 (대시보드 기본값)
LONG_RSI_OVERSOLD = 35
LONG_RSI_EXIT = 40
LONG_RSI_OVERBOUGHT = 80
LONG_RSI_SELL = 55
LONG_STOP_LOSS = -25

SHORT_RSI_PEAK = 78
SHORT_RSI_ENTRY = 65
SHORT_RSI_EXIT = 45
SHORT_STOP_LOSS = -15
SHORT_MAX_HOLD = 42
SHORT_LOOKBACK = 24
DC_RSI_THRESHOLD = 55
SHORT_MAX_ENTRIES = 4


def run_test(df, name):
    """테스트 실행"""
    long_signals = find_long_signals(df, LONG_RSI_OVERSOLD, LONG_RSI_EXIT, True)
    long_exit_signals = find_long_exit_signals(df, LONG_RSI_OVERBOUGHT, LONG_RSI_SELL)
    short_signals = find_short_signals(df, SHORT_RSI_PEAK, SHORT_RSI_ENTRY, SHORT_LOOKBACK, DC_RSI_THRESHOLD)
    short_exit_signals = find_short_exit_signals(df, LONG_RSI_OVERSOLD, SHORT_RSI_EXIT)
    
    trades, _ = simulate_dual_trades(
        df, long_signals, long_exit_signals,
        short_signals, short_exit_signals,
        LONG_STOP_LOSS, SHORT_STOP_LOSS, SHORT_MAX_HOLD, SHORT_MAX_ENTRIES
    )
    
    long_trades = [t for t in trades if t['type'] == 'long']
    short_trades = [t for t in trades if t['type'] == 'short']
    
    total_return = sum(t['return'] for t in trades)
    long_return = sum(t['return'] for t in long_trades)
    short_return = sum(t['return'] for t in short_trades)
    
    long_wins = len([t for t in long_trades if t['return'] > 0])
    short_wins = len([t for t in short_trades if t['return'] > 0])
    
    return {
        'name': name,
        'total': total_return,
        'long': long_return,
        'short': short_return,
        'total_trades': len(trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win': long_wins / len(long_trades) * 100 if long_trades else 0,
        'short_win': short_wins / len(short_trades) * 100 if short_trades else 0
    }


# ===== 1. 대시보드 데이터 (yfinance) =====
print("\n📊 1. 대시보드 데이터 (yfinance)")
df_dashboard = load_data('BTC-USD')
print(f"   기간: {df_dashboard.index[0]} ~ {df_dashboard.index[-1]}")
print(f"   봉 수: {len(df_dashboard)}개")

result_dashboard = run_test(df_dashboard, "대시보드(yfinance)")

# ===== 2. 새로 수집한 데이터 (OKX) =====
print("\n📊 2. 새 데이터 (OKX)")
df_okx = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
print(f"   전체: {df_okx.index[0]} ~ {df_okx.index[-1]} ({len(df_okx)}봉)")

# 대시보드와 동일한 기간으로 필터링
start_date = df_dashboard.index[0]
end_date = df_dashboard.index[-1]

df_okx_filtered = df_okx[(df_okx.index >= start_date) & (df_okx.index <= end_date)]
print(f"   필터링: {df_okx_filtered.index[0]} ~ {df_okx_filtered.index[-1]} ({len(df_okx_filtered)}봉)")

result_okx = run_test(df_okx_filtered, "새데이터(OKX)")

# ===== 비교 =====
print("\n" + "=" * 100)
print("📊 검증 결과")
print("=" * 100)

print(f"\n{'항목':<15} | {'대시보드(yfinance)':>20} | {'새데이터(OKX)':>20} | {'일치':>6}")
print("-" * 75)

def check(a, b, tol=0.5):
    return "✅" if abs(a - b) < tol else "❌"

print(f"{'총 수익률':<15} | {result_dashboard['total']:>+19.1f}% | {result_okx['total']:>+19.1f}% | {check(result_dashboard['total'], result_okx['total'], 1):>6}")
print(f"{'롱 수익률':<15} | {result_dashboard['long']:>+19.1f}% | {result_okx['long']:>+19.1f}% | {check(result_dashboard['long'], result_okx['long'], 1):>6}")
print(f"{'숏 수익률':<15} | {result_dashboard['short']:>+19.1f}% | {result_okx['short']:>+19.1f}% | {check(result_dashboard['short'], result_okx['short'], 1):>6}")
print(f"{'총 거래':<15} | {result_dashboard['total_trades']:>20} | {result_okx['total_trades']:>20} | {check(result_dashboard['total_trades'], result_okx['total_trades'], 0):>6}")
print(f"{'롱 거래':<15} | {result_dashboard['long_trades']:>20} | {result_okx['long_trades']:>20} | {check(result_dashboard['long_trades'], result_okx['long_trades'], 0):>6}")
print(f"{'숏 거래':<15} | {result_dashboard['short_trades']:>20} | {result_okx['short_trades']:>20} | {check(result_dashboard['short_trades'], result_okx['short_trades'], 0):>6}")
print(f"{'롱 승률':<15} | {result_dashboard['long_win']:>19.1f}% | {result_okx['long_win']:>19.1f}% | {check(result_dashboard['long_win'], result_okx['long_win'], 1):>6}")
print(f"{'숏 승률':<15} | {result_dashboard['short_win']:>19.1f}% | {result_okx['short_win']:>19.1f}% | {check(result_dashboard['short_win'], result_okx['short_win'], 1):>6}")

print("\n" + "=" * 100)
print("📌 대시보드 목표값: 총 43회, +145.1%, 롱 18회 +112.4%, 숏 25회 +32.8%")
print("=" * 100)

