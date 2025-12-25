"""
골든크로스 ON/OFF 연도별 비교
하락장(2022년)에서 어떤 게 더 좋은지 확인
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
print("🔍 골든크로스 ON/OFF 연도별 비교")
print("=" * 100)

# 파라미터
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25


def run_test_yearly(df, use_gc, name):
    buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, use_gc)
    sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
    trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)
    
    # 연도별 분석
    yearly = {}
    for trade in trades:
        year = trade['exit_date'].year
        if year not in yearly:
            yearly[year] = {'return': 0, 'count': 0, 'wins': 0}
        yearly[year]['return'] += trade['return']
        yearly[year]['count'] += 1
        if trade['return'] > 0:
            yearly[year]['wins'] += 1
    
    total_return = sum(t['return'] for t in trades)
    
    return {
        'name': name,
        'total': total_return,
        'trades': len(trades),
        'yearly': yearly
    }


# ===== OKX 5년 데이터 =====
print("\n📊 OKX 5년 데이터 로드...")
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()
print(f"   기간: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")

result_on = run_test_yearly(df, True, "GC ON")
result_off = run_test_yearly(df, False, "GC OFF")

# 연도별 비교
print("\n" + "=" * 100)
print("📅 연도별 수익률 비교")
print("=" * 100)

all_years = sorted(set(result_on['yearly'].keys()) | set(result_off['yearly'].keys()))

print(f"\n{'연도':>6} | {'GC ON':>12} | {'GC OFF':>12} | {'차이':>10} | {'더 좋은 것':>10}")
print("-" * 65)

for year in all_years:
    on_data = result_on['yearly'].get(year, {'return': 0, 'count': 0})
    off_data = result_off['yearly'].get(year, {'return': 0, 'count': 0})
    
    on_ret = on_data['return']
    off_ret = off_data['return']
    diff = off_ret - on_ret
    better = "GC OFF" if off_ret > on_ret else "GC ON" if on_ret > off_ret else "동일"
    
    print(f"{year:>6} | {on_ret:>+11.1f}% | {off_ret:>+11.1f}% | {diff:>+9.1f}% | {better:>10}")

print("-" * 65)
print(f"{'합계':>6} | {result_on['total']:>+11.1f}% | {result_off['total']:>+11.1f}% | {result_off['total'] - result_on['total']:>+9.1f}% |")

# 거래 횟수 비교
print("\n" + "=" * 100)
print("📊 연도별 거래 횟수 비교")
print("=" * 100)

print(f"\n{'연도':>6} | {'GC ON':>8} | {'GC OFF':>8} | {'차이':>6}")
print("-" * 40)

for year in all_years:
    on_data = result_on['yearly'].get(year, {'count': 0})
    off_data = result_off['yearly'].get(year, {'count': 0})
    
    print(f"{year:>6} | {on_data['count']:>8} | {off_data['count']:>8} | {off_data['count'] - on_data['count']:>+6}")

print("-" * 40)
print(f"{'합계':>6} | {result_on['trades']:>8} | {result_off['trades']:>8} | {result_off['trades'] - result_on['trades']:>+6}")

# 핵심 분석
print("\n" + "=" * 100)
print("🔍 핵심 분석")
print("=" * 100)

# 2022년 (하락장) 상세
if 2022 in result_on['yearly'] and 2022 in result_off['yearly']:
    on_2022 = result_on['yearly'][2022]
    off_2022 = result_off['yearly'][2022]
    
    print(f"\n📉 2022년 하락장:")
    print(f"   GC ON:  {on_2022['return']:+.1f}% ({on_2022['count']}회)")
    print(f"   GC OFF: {off_2022['return']:+.1f}% ({off_2022['count']}회)")
    print(f"   차이: {off_2022['return'] - on_2022['return']:+.1f}%")
    
    if on_2022['return'] > off_2022['return']:
        print(f"\n   ⚠️ 하락장에서는 GC ON이 {on_2022['return'] - off_2022['return']:+.1f}% 더 좋음!")
    else:
        print(f"\n   ✅ 하락장에서도 GC OFF가 더 좋음")

print("\n" + "=" * 100)
print("✅ 분석 완료!")
print("=" * 100)

