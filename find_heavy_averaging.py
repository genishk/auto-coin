"""
물타기 많이 한 시기 찾기
현재 대시보드 전략 (GC OFF) 기준
"""
import sys
sys.path.insert(0, '.')

import pandas as pd

from dashboard_4h import (
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

print("=" * 100)
print("📊 물타기 많이 한 시기 분석 (GC OFF)")
print("=" * 100)

# 파라미터 (GC OFF)
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False

# 데이터 로드
print("\n📊 데이터 로드...")
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()
print(f"   기간: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")

# 시그널 및 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

print(f"   총 거래: {len(trades)}회")

# 물타기 많은 거래 찾기 (5회 이상)
heavy_trades = [t for t in trades if t['num_buys'] >= 5]
print(f"   물타기 5회 이상: {len(heavy_trades)}회")

# ===== 물타기 많은 거래 상세 =====
print("\n" + "=" * 100)
print("🔴 물타기 5회 이상 거래 (리스크 높았던 시기)")
print("=" * 100)

print(f"\n{'진입일':>12} | {'청산일':>12} | {'물타기':>6} | {'수익률':>10} | {'청산사유':>8} | {'진입가':>12} | {'청산가':>12}")
print("-" * 90)

for trade in sorted(heavy_trades, key=lambda x: x['entry_dates'][0]):
    entry_date = trade['entry_dates'][0].strftime('%Y-%m-%d')
    exit_date = trade['exit_date'].strftime('%Y-%m-%d')
    num_buys = trade['num_buys']
    ret = trade['return']
    reason = trade['exit_reason']
    avg_price = trade['avg_price']
    exit_price = trade['exit_price']
    
    print(f"{entry_date:>12} | {exit_date:>12} | {num_buys:>5}회 | {ret:>+9.1f}% | {reason:>8} | ${avg_price:>10,.0f} | ${exit_price:>10,.0f}")

# ===== 연도별 물타기 분포 =====
print("\n" + "=" * 100)
print("📅 연도별 물타기 분포")
print("=" * 100)

yearly_stats = {}
for trade in trades:
    year = trade['entry_dates'][0].year
    if year not in yearly_stats:
        yearly_stats[year] = {'trades': 0, 'total_buys': 0, 'heavy': 0, 'loss_trades': 0}
    yearly_stats[year]['trades'] += 1
    yearly_stats[year]['total_buys'] += trade['num_buys']
    if trade['num_buys'] >= 5:
        yearly_stats[year]['heavy'] += 1
    if trade['return'] < 0:
        yearly_stats[year]['loss_trades'] += 1

print(f"\n{'연도':>6} | {'거래':>6} | {'총물타기':>8} | {'평균물타기':>10} | {'5회이상':>8} | {'손실거래':>8}")
print("-" * 65)

for year in sorted(yearly_stats.keys()):
    s = yearly_stats[year]
    avg_buys = s['total_buys'] / s['trades'] if s['trades'] > 0 else 0
    print(f"{year:>6} | {s['trades']:>6} | {s['total_buys']:>8} | {avg_buys:>9.1f}회 | {s['heavy']:>8} | {s['loss_trades']:>8}")

# ===== 물타기 많은 시기 요약 =====
print("\n" + "=" * 100)
print("📌 물타기 많았던 시기 (리스크 헷징 필요 시기)")
print("=" * 100)

print("\n물타기 5회 이상 거래가 발생한 기간:")
for trade in sorted(heavy_trades, key=lambda x: x['entry_dates'][0]):
    entry_start = trade['entry_dates'][0].strftime('%Y-%m-%d')
    entry_end = trade['entry_dates'][-1].strftime('%Y-%m-%d')
    exit_date = trade['exit_date'].strftime('%Y-%m-%d')
    duration = (trade['exit_date'] - trade['entry_dates'][0]).days
    
    print(f"  📍 {entry_start} ~ {exit_date} ({duration}일간, 물타기 {trade['num_buys']}회, {trade['exit_reason']} {trade['return']:+.1f}%)")

print("\n" + "=" * 100)
print("✅ 분석 완료!")
print("=" * 100)

