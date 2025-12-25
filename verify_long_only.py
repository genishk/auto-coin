"""
롱 전용 대시보드(dashboard_4h.py) 전략 검증
yfinance 데이터 vs OKX 데이터 비교
"""
import sys
sys.path.insert(0, '.')

import pandas as pd

# dashboard_4h.py 함수 직접 import
from dashboard_4h import (
    load_data,
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

print("=" * 100)
print("🔍 롱 전용 대시보드 전략 검증 (yfinance vs OKX)")
print("=" * 100)

# 대시보드 기본 파라미터
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = True


def run_test(df, name):
    """대시보드 함수 그대로 사용"""
    buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
    sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
    trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)
    
    total_return = sum(t['return'] for t in trades)
    wins = len([t for t in trades if t['return'] > 0])
    win_rate = wins / len(trades) * 100 if trades else 0
    avg_return = total_return / len(trades) if trades else 0
    
    return {
        'name': name,
        'total': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'trade_list': trades
    }


# ===== 1. yfinance 데이터 (대시보드와 동일) =====
print("\n📊 1. yfinance 데이터 (대시보드)")
df_yf = load_data('BTC-USD')
print(f"   기간: {df_yf.index[0]} ~ {df_yf.index[-1]} ({len(df_yf)}봉)")

result_yf = run_test(df_yf, "yfinance")
print(f"   거래: {result_yf['trades']}회, 승률: {result_yf['win_rate']:.0f}%")
print(f"   평균: {result_yf['avg_return']:+.1f}%, 총수익: {result_yf['total']:+.1f}%")

# ===== 2. OKX 데이터 (동일 기간) =====
print("\n📊 2. OKX 데이터 (동일 기간)")
df_okx_full = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)

# 동일 기간으로 필터링
start_date = df_yf.index[0]
end_date = df_yf.index[-1]
df_okx = df_okx_full[(df_okx_full.index >= start_date) & (df_okx_full.index <= end_date)]
print(f"   기간: {df_okx.index[0]} ~ {df_okx.index[-1]} ({len(df_okx)}봉)")

result_okx = run_test(df_okx, "OKX")
print(f"   거래: {result_okx['trades']}회, 승률: {result_okx['win_rate']:.0f}%")
print(f"   평균: {result_okx['avg_return']:+.1f}%, 총수익: {result_okx['total']:+.1f}%")

# ===== 비교 =====
print("\n" + "=" * 100)
print("📊 검증 결과")
print("=" * 100)

print(f"\n{'항목':<15} | {'yfinance':>15} | {'OKX':>15} | {'일치':>6}")
print("-" * 60)
print(f"{'총 거래':<15} | {result_yf['trades']:>15} | {result_okx['trades']:>15} | {'✅' if result_yf['trades'] == result_okx['trades'] else '❌':>6}")
print(f"{'승률':<15} | {result_yf['win_rate']:>14.0f}% | {result_okx['win_rate']:>14.0f}% | {'✅' if abs(result_yf['win_rate'] - result_okx['win_rate']) < 5 else '❌':>6}")
print(f"{'평균 수익률':<15} | {result_yf['avg_return']:>+14.1f}% | {result_okx['avg_return']:>+14.1f}% | {'✅' if abs(result_yf['avg_return'] - result_okx['avg_return']) < 1 else '❌':>6}")
print(f"{'총 수익률':<15} | {result_yf['total']:>+14.1f}% | {result_okx['total']:>+14.1f}% | {'✅' if abs(result_yf['total'] - result_okx['total']) < 10 else '❌':>6}")

print("\n📌 대시보드 목표값: 22회, 승률 95%, 평균 +5.3%, 총수익 +115.7%")

# ===== 5년치 OKX 데이터 테스트 =====
print("\n" + "=" * 100)
print("📊 5년치 OKX 데이터 테스트 (롱 전용)")
print("=" * 100)

df_okx_5y = df_okx_full.dropna()
print(f"   기간: {df_okx_5y.index[0]} ~ {df_okx_5y.index[-1]} ({len(df_okx_5y)}봉)")

result_5y = run_test(df_okx_5y, "OKX 5년")
print(f"   거래: {result_5y['trades']}회, 승률: {result_5y['win_rate']:.0f}%")
print(f"   평균: {result_5y['avg_return']:+.1f}%, 총수익: {result_5y['total']:+.1f}%")

# 연도별 분석
print("\n📅 연도별 성과:")
yearly = {}
for trade in result_5y['trade_list']:
    year = trade['exit_date'].year
    if year not in yearly:
        yearly[year] = {'return': 0, 'count': 0}
    yearly[year]['return'] += trade['return']
    yearly[year]['count'] += 1

print(f"{'연도':>6} | {'수익률':>10} | {'거래':>6}")
print("-" * 30)
for year in sorted(yearly.keys()):
    y = yearly[year]
    print(f"{year:>6} | {y['return']:>+9.1f}% | {y['count']:>6}")

print("\n" + "=" * 100)
print("✅ 검증 완료!")
print("=" * 100)

