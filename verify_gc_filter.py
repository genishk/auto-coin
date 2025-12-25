"""
골든크로스 필터 ON/OFF 비교 (2년 데이터)
대시보드 결과와 일치하는지 확인
"""
import sys
sys.path.insert(0, '.')

import pandas as pd

from dashboard_4h import (
    load_data,
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

print("=" * 100)
print("🔍 골든크로스 필터 ON/OFF 비교 (2년 데이터)")
print("=" * 100)

# 파라미터 (대시보드 기본값)
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25


def run_test(df, use_gc, name):
    buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, use_gc)
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
        'avg_return': avg_return
    }


# ===== yfinance 데이터 (대시보드 원본) =====
print("\n📊 yfinance 데이터 (대시보드)")
df_yf = load_data('BTC-USD')
print(f"   기간: {df_yf.index[0]} ~ {df_yf.index[-1]} ({len(df_yf)}봉)")

result_yf_on = run_test(df_yf, True, "yfinance GC ON")
result_yf_off = run_test(df_yf, False, "yfinance GC OFF")

print(f"\n{'설정':<20} | {'거래':>6} | {'승률':>8} | {'평균':>10} | {'총수익':>12}")
print("-" * 70)
print(f"{'골든크로스 ON':<20} | {result_yf_on['trades']:>6} | {result_yf_on['win_rate']:>7.0f}% | {result_yf_on['avg_return']:>+9.1f}% | {result_yf_on['total']:>+11.1f}%")
print(f"{'골든크로스 OFF':<20} | {result_yf_off['trades']:>6} | {result_yf_off['win_rate']:>7.0f}% | {result_yf_off['avg_return']:>+9.1f}% | {result_yf_off['total']:>+11.1f}%")

print("\n📌 대시보드 목표값: 22회, 95%, +5.3%, +115.7%")

# ===== OKX 데이터 (동일 기간) =====
print("\n" + "=" * 100)
print("📊 OKX 데이터 (동일 2년 기간)")
print("=" * 100)

df_okx_full = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
start_date = df_yf.index[0]
end_date = df_yf.index[-1]
df_okx = df_okx_full[(df_okx_full.index >= start_date) & (df_okx_full.index <= end_date)]
print(f"   기간: {df_okx.index[0]} ~ {df_okx.index[-1]} ({len(df_okx)}봉)")

result_okx_on = run_test(df_okx, True, "OKX GC ON")
result_okx_off = run_test(df_okx, False, "OKX GC OFF")

print(f"\n{'설정':<20} | {'거래':>6} | {'승률':>8} | {'평균':>10} | {'총수익':>12}")
print("-" * 70)
print(f"{'골든크로스 ON':<20} | {result_okx_on['trades']:>6} | {result_okx_on['win_rate']:>7.0f}% | {result_okx_on['avg_return']:>+9.1f}% | {result_okx_on['total']:>+11.1f}%")
print(f"{'골든크로스 OFF':<20} | {result_okx_off['trades']:>6} | {result_okx_off['win_rate']:>7.0f}% | {result_okx_off['avg_return']:>+9.1f}% | {result_okx_off['total']:>+11.1f}%")

# ===== 5년 OKX 데이터 =====
print("\n" + "=" * 100)
print("📊 OKX 데이터 (5년 전체)")
print("=" * 100)

df_okx_5y = df_okx_full.dropna()
print(f"   기간: {df_okx_5y.index[0]} ~ {df_okx_5y.index[-1]} ({len(df_okx_5y)}봉)")

result_5y_on = run_test(df_okx_5y, True, "OKX 5Y GC ON")
result_5y_off = run_test(df_okx_5y, False, "OKX 5Y GC OFF")

print(f"\n{'설정':<20} | {'거래':>6} | {'승률':>8} | {'평균':>10} | {'총수익':>12}")
print("-" * 70)
print(f"{'골든크로스 ON':<20} | {result_5y_on['trades']:>6} | {result_5y_on['win_rate']:>7.0f}% | {result_5y_on['avg_return']:>+9.1f}% | {result_5y_on['total']:>+11.1f}%")
print(f"{'골든크로스 OFF':<20} | {result_5y_off['trades']:>6} | {result_5y_off['win_rate']:>7.0f}% | {result_5y_off['avg_return']:>+9.1f}% | {result_5y_off['total']:>+11.1f}%")

print("\n" + "=" * 100)
print("✅ 비교 완료!")
print("=" * 100)

