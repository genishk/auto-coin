"""
2022-2023년 하락장 시뮬레이션
대시보드와 완전히 동일한 로직 사용
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# 대시보드 함수 임포트
from dashboard_4h import find_buy_signals, find_sell_signals, simulate_trades

def add_indicators(df):
    """지표 추가 (대시보드와 동일)"""
    df = df.copy()
    
    # RSI 계산
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

# 5년 4시간봉 데이터 로드
df = pd.read_csv('data/btc_4h_5y.csv', index_col=0, parse_dates=True)
print(f"전체 데이터: {df.index[0]} ~ {df.index[-1]}")
print(f"총 {len(df)} 캔들")

# 2022-2023년 필터링 (지표 계산을 위해 앞부분 데이터 포함)
df_full = df[df.index <= '2023-12-31'].copy()
df_full = add_indicators(df_full)

# 2022-2023년만 추출
df_2022_2023 = df_full[df_full.index >= '2022-01-01'].copy()
print(f"\n2022-2023년 데이터: {df_2022_2023.index[0]} ~ {df_2022_2023.index[-1]}")
print(f"총 {len(df_2022_2023)} 캔들")

# 대시보드 기본 설정값
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False  # OFF 권장

# 헷징 설정 (새 전략)
USE_HEDGE = True
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE_INTERVAL = 3
HEDGE_RATIO = 1.0  # 100%
HEDGE_PROFIT = 8
HEDGE_STOP = -15

CAPITAL_PER_ENTRY = 1000

# 시그널 생성
buy_signals = find_buy_signals(df_2022_2023, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df_2022_2023, RSI_OVERBOUGHT, RSI_SELL_EXIT)

print(f"\n매수 시그널: {len(buy_signals)}회")
print(f"매도 시그널: {len(sell_signals)}회")

# 시뮬레이션 실행
trades, current_positions, hedge_trades, current_hedge = simulate_trades(
    df_2022_2023, buy_signals, sell_signals, STOP_LOSS,
    use_hedge=USE_HEDGE, hedge_threshold=HEDGE_THRESHOLD,
    hedge_upgrade_interval=HEDGE_UPGRADE_INTERVAL, hedge_ratio=HEDGE_RATIO,
    hedge_profit=HEDGE_PROFIT, hedge_stop=HEDGE_STOP
)

print("\n" + "="*80)
print("📊 2022-2023년 시뮬레이션 결과 (대시보드와 동일한 로직)")
print("="*80)

# 롱 성과
long_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in trades)
long_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in trades)

print(f"\n🟢 롱 성과:")
print(f"   완료 거래: {len(trades)}회")
print(f"   총 투자금: ${long_invested:,.0f}")
print(f"   총 손익: ${long_profit:+,.0f}")
if long_invested > 0:
    print(f"   수익률: {long_profit/long_invested*100:+.1f}%")

wins = len([t for t in trades if t['return'] > 0])
if trades:
    print(f"   승률: {wins/len(trades)*100:.0f}%")

# 거래별 상세
print(f"\n   📋 거래 상세:")
for i, t in enumerate(trades, 1):
    profit = t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100
    print(f"      {i}. {t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}")
    print(f"         물타기: {t['num_buys']}회, 투자금: ${t['num_buys']*CAPITAL_PER_ENTRY:,}, 수익률: {t['return']:+.1f}%, 손익: ${profit:+,.0f}, 사유: {t['exit_reason']}")

# 숏 헷징 성과
if hedge_trades:
    short_invested = sum(h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY) for h in hedge_trades)
    short_profit = sum(h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY) * h['return'] / 100 for h in hedge_trades)
    
    print(f"\n🟣 숏 헷징 성과:")
    print(f"   헷징 발동: {len(hedge_trades)}회")
    print(f"   총 투자금: ${short_invested:,.0f}")
    print(f"   총 손익: ${short_profit:+,.0f}")
    if short_invested > 0:
        print(f"   수익률: {short_profit/short_invested*100:+.1f}%")
    
    hedge_wins = len([h for h in hedge_trades if h['return'] > 0])
    print(f"   승률: {hedge_wins/len(hedge_trades)*100:.0f}%")
    
    print(f"\n   📋 헷징 상세:")
    for i, h in enumerate(hedge_trades, 1):
        invested = h.get('invested', h['long_num_buys'] * CAPITAL_PER_ENTRY)
        profit = invested * h['return'] / 100
        print(f"      {i}. {h['entry_date'].strftime('%Y-%m-%d')} ~ {h['exit_date'].strftime('%Y-%m-%d')}")
        print(f"         투자금: ${invested:,}, 수익률: {h['return']:+.1f}%, 손익: ${profit:+,.0f}, 사유: {h['exit_reason']}")
else:
    short_profit = 0
    print(f"\n🟣 숏 헷징: 발동 없음")

# 총 성과
total_profit = long_profit + short_profit
print(f"\n💰 총 성과:")
print(f"   롱 손익: ${long_profit:+,.0f}")
print(f"   숏 손익: ${short_profit:+,.0f}")
print(f"   ─────────────────")
print(f"   총 손익: ${total_profit:+,.0f}")
if long_invested > 0:
    print(f"   총 수익률 (롱 투자금 대비): {total_profit/long_invested*100:+.1f}%")

# 미청산 포지션
if current_positions:
    print(f"\n⚠️ 2023년 말 미청산 롱 포지션:")
    print(f"   물타기: {len(current_positions)}회")
    total_qty = sum(1/p['price'] for p in current_positions)
    avg_price = len(current_positions) / total_qty
    last_price = df_2022_2023['Close'].iloc[-1]
    unrealized = (last_price / avg_price - 1) * 100
    print(f"   평단가: ${avg_price:,.2f}")
    print(f"   2023년 말 가격: ${last_price:,.2f}")
    print(f"   미실현 수익률: {unrealized:+.1f}%")

if current_hedge:
    print(f"\n⚠️ 2023년 말 미청산 숏 포지션:")
    last_price = df_2022_2023['Close'].iloc[-1]
    short_return = (current_hedge['entry_price'] - last_price) / current_hedge['entry_price'] * 100
    print(f"   진입가: ${current_hedge['entry_price']:,.2f}")
    print(f"   2023년 말 가격: ${last_price:,.2f}")
    print(f"   미실현 수익률: {short_return:+.1f}%")

# 비트코인 가격 변동
btc_start = df_2022_2023['Close'].iloc[0]
btc_end = df_2022_2023['Close'].iloc[-1]
btc_change = (btc_end / btc_start - 1) * 100
print(f"\n📈 비트코인 2022-2023년:")
print(f"   2022년 초: ${btc_start:,.0f}")
print(f"   2023년 말: ${btc_end:,.0f}")
print(f"   변동: {btc_change:+.1f}%")

