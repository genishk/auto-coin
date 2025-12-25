"""
2022-2025 년별 롱/숏 수익 분석
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from dashboard_4h import find_buy_signals, find_sell_signals, simulate_trades

def add_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    return df

def test_period(df):
    buy_signals = find_buy_signals(df, 35, 40, False)
    sell_signals = find_sell_signals(df, 80, 55)
    
    trades, _, hedge_trades, _ = simulate_trades(
        df, buy_signals, sell_signals, -25,
        use_hedge=True, hedge_threshold=2,
        hedge_upgrade_interval=3, hedge_ratio=1.0,
        hedge_profit=8, hedge_stop=-15
    )
    
    CAPITAL = 1000
    
    long_invested = sum(t['num_buys'] * CAPITAL for t in trades)
    long_profit = sum(t['num_buys'] * CAPITAL * t['return'] / 100 for t in trades)
    long_wins = len([t for t in trades if t['return'] > 0])
    
    if hedge_trades:
        short_invested = sum(h.get('invested', h['long_num_buys'] * CAPITAL) for h in hedge_trades)
        short_profit = sum(h.get('invested', h['long_num_buys'] * CAPITAL) * h['return'] / 100 for h in hedge_trades)
        short_wins = len([h for h in hedge_trades if h['return'] > 0])
        short_count = len(hedge_trades)
    else:
        short_invested, short_profit, short_wins, short_count = 0, 0, 0, 0
    
    return {
        'long_trades': len(trades),
        'long_wins': long_wins,
        'long_invested': long_invested,
        'long_profit': long_profit,
        'short_count': short_count,
        'short_wins': short_wins,
        'short_invested': short_invested,
        'short_profit': short_profit,
        'total_profit': long_profit + short_profit
    }

# 데이터 로드
df_full = pd.read_csv('data/btc_4h_5y.csv', index_col=0, parse_dates=True)
df_full = add_indicators(df_full)

years = [2022, 2023, 2024, 2025]

print("="*100)
print("📊 년별 롱/숏 수익 분석 (2022-2025)")
print("="*100)
print(f"설정: RSI 35/40/80/55, 손절 -25%, 헷징 2회 시작/3회 업그레이드/100%/익절 8%/손절 -15%")
print("="*100)

results = []

for year in years:
    start = f'{year}-01-01'
    end = f'{year}-12-31' if year < 2025 else df_full.index[-1]
    
    df_year = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()
    
    if len(df_year) < 100:
        continue
    
    btc_start = df_year['Close'].iloc[0]
    btc_end = df_year['Close'].iloc[-1]
    btc_change = (btc_end / btc_start - 1) * 100
    
    r = test_period(df_year)
    r['year'] = year
    r['btc_start'] = btc_start
    r['btc_end'] = btc_end
    r['btc_change'] = btc_change
    results.append(r)

# 헤더
print(f"\n{'년도':^6} | {'BTC변동':^12} | {'롱거래':^6} | {'롱승률':^6} | {'롱손익':^12} | {'숏거래':^6} | {'숏승률':^6} | {'숏손익':^12} | {'총손익':^12}")
print("-" * 110)

total_long = 0
total_short = 0

for r in results:
    year = r['year']
    btc = f"{r['btc_change']:+.0f}%"
    long_wr = f"{r['long_wins']/r['long_trades']*100:.0f}%" if r['long_trades'] > 0 else "N/A"
    short_wr = f"{r['short_wins']/r['short_count']*100:.0f}%" if r['short_count'] > 0 else "N/A"
    
    total_long += r['long_profit']
    total_short += r['short_profit']
    
    print(f" {year:^5} | {btc:^12} | {r['long_trades']:^6} | {long_wr:^6} | ${r['long_profit']:>+10,.0f} | {r['short_count']:^6} | {short_wr:^6} | ${r['short_profit']:>+10,.0f} | ${r['total_profit']:>+10,.0f}")

print("-" * 110)
print(f" {'합계':^5} | {'-':^12} | {'-':^6} | {'-':^6} | ${total_long:>+10,.0f} | {'-':^6} | {'-':^6} | ${total_short:>+10,.0f} | ${total_long+total_short:>+10,.0f}")

# 상세 분석
print("\n" + "="*100)
print("📈 년별 상세 분석")
print("="*100)

for r in results:
    year = r['year']
    
    # 시장 상황 판단
    if r['btc_change'] < -30:
        market = "📉 하락장"
    elif r['btc_change'] < 30:
        market = "➡️ 횡보장"
    else:
        market = "📈 상승장"
    
    print(f"\n{'─'*50}")
    print(f"📅 {year}년 ({market})")
    print(f"{'─'*50}")
    print(f"  BTC: ${r['btc_start']:,.0f} → ${r['btc_end']:,.0f} ({r['btc_change']:+.0f}%)")
    print()
    print(f"  🟢 롱: {r['long_trades']}회 거래, 승률 {r['long_wins']/r['long_trades']*100:.0f}%")
    print(f"      투자금 ${r['long_invested']:,.0f} → 손익 ${r['long_profit']:+,.0f}")
    if r['long_invested'] > 0:
        print(f"      수익률: {r['long_profit']/r['long_invested']*100:+.1f}%")
    print()
    print(f"  🟣 숏: {r['short_count']}회 거래, 승률 {r['short_wins']/r['short_count']*100:.0f}%" if r['short_count'] > 0 else "  🟣 숏: 발동 없음")
    if r['short_count'] > 0:
        print(f"      투자금 ${r['short_invested']:,.0f} → 손익 ${r['short_profit']:+,.0f}")
        if r['short_invested'] > 0:
            print(f"      수익률: {r['short_profit']/r['short_invested']*100:+.1f}%")
    print()
    print(f"  💰 총: ${r['total_profit']:+,.0f}")
    
    # 숏 효과 분석
    if r['short_profit'] > 0:
        print(f"  ✅ 숏 효과: +${r['short_profit']:,.0f} 추가 수익!")
    else:
        print(f"  ⚠️ 숏 효과: ${r['short_profit']:,.0f} (롱만 했으면 ${r['long_profit']:+,.0f})")

