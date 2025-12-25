"""
대시보드 180일 결과 검증
대시보드와 동일한 계산 방식인지 확인
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')

from dashboard_4h import find_buy_signals, find_sell_signals, simulate_trades

def add_indicators(df):
    """지표 추가"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MA
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# 데이터 로드
df = pd.read_csv('data/btc_4h_5y.csv', index_col=0, parse_dates=True)
df = add_indicators(df)

print(f"데이터 범위: {df.index[0]} ~ {df.index[-1]}")

# 대시보드 설정값 (현재 대시보드와 동일)
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False

# 헷징 설정 (대시보드 기본값)
USE_HEDGE = True
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE_INTERVAL = 3
HEDGE_RATIO = 1.0  # 100%
HEDGE_PROFIT = 8
HEDGE_STOP = -15

CAPITAL_PER_ENTRY = 1000

# 시그널 생성
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)

# 시뮬레이션
trades, current_positions, hedge_trades, current_hedge = simulate_trades(
    df, buy_signals, sell_signals, STOP_LOSS,
    use_hedge=USE_HEDGE, hedge_threshold=HEDGE_THRESHOLD,
    hedge_upgrade_interval=HEDGE_UPGRADE_INTERVAL, hedge_ratio=HEDGE_RATIO,
    hedge_profit=HEDGE_PROFIT, hedge_stop=HEDGE_STOP
)

# 180일 필터링 (대시보드와 동일)
lookback_days = 180
# 4시간봉이므로 하루 6개 캔들
signal_cutoff = df.index[-1] - timedelta(days=lookback_days)

print(f"\n180일 기준: {signal_cutoff} 이후")

# 필터링
filtered_trades = [t for t in trades if t['exit_date'] >= signal_cutoff]
filtered_hedges = [h for h in hedge_trades if h['exit_date'] >= signal_cutoff]

print(f"\n" + "="*60)
print("📊 대시보드 180일 결과 검증")
print("="*60)

# 롱 성과
long_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in filtered_trades)
long_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in filtered_trades)

print(f"\n🟢 롱 성과:")
print(f"   롱 거래: {len(filtered_trades)}회")
wins = len([t for t in filtered_trades if t['return'] > 0])
if filtered_trades:
    print(f"   승률: {wins/len(filtered_trades)*100:.0f}%")
print(f"   롱 손익: ${long_profit:+,.0f}")
if long_invested > 0:
    print(f"   롱 수익률: {long_profit/long_invested*100:+.1f}%")

# 숏 헷징 성과
if filtered_hedges:
    short_invested = sum(h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY) for h in filtered_hedges)
    short_profit = sum(h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY) * h['return'] / 100 for h in filtered_hedges)
    hedge_wins = len([h for h in filtered_hedges if h['return'] > 0])
    
    print(f"\n🔴 숏 헷징 성과:")
    print(f"   헷징 발동: {len(filtered_hedges)}회")
    print(f"   숏 승률: {hedge_wins/len(filtered_hedges)*100:.0f}%")
    print(f"   숏 손익: ${short_profit:+,.0f}")
    if short_invested > 0:
        print(f"   숏 수익률: {short_profit/short_invested*100:+.1f}%")
else:
    short_profit = 0
    print(f"\n🔴 숏 헷징: 발동 없음")

# 총 성과
total_profit = long_profit + short_profit
print(f"\n💰 총 성과:")
print(f"   총 투자금: ${long_invested:,.0f}")
print(f"   총 손익: ${total_profit:+,.0f}")
if long_invested > 0:
    print(f"   금액 수익률: {total_profit/long_invested*100:+.2f}%")
print(f"   숏 헷징 효과: ${short_profit:+,.0f}")

# 상세 거래 내역
print(f"\n📋 롱 거래 상세:")
for i, t in enumerate(filtered_trades, 1):
    profit = t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100
    print(f"   {i}. {t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}")
    print(f"      물타기: {t['num_buys']}회, 수익률: {t['return']:+.1f}%, 손익: ${profit:+,.0f}")

print(f"\n📋 숏 헷징 상세:")
for i, h in enumerate(filtered_hedges, 1):
    invested = h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY)
    profit = invested * h['return'] / 100
    print(f"   {i}. {h['entry_date'].strftime('%Y-%m-%d')} ~ {h['exit_date'].strftime('%Y-%m-%d')}")
    print(f"      투자금: ${invested:,.0f}, 수익률: {h['return']:+.1f}%, 손익: ${profit:+,.0f}, 사유: {h['exit_reason']}")

