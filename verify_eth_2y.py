"""
ETH 최근 2년 결과 검증
- OKX 5년 데이터 vs 대시보드(yfinance) 비교
"""
import pandas as pd
import numpy as np
from datetime import timedelta

# 5년 데이터로 최근 2년 테스트
df = pd.read_csv('data/eth_4h_5y.csv', index_col='Date', parse_dates=True)

# 최근 730일
two_years = df.index[-1] - timedelta(days=730)
df_2y = df[df.index >= two_years].copy()

print('=' * 60)
print('ETH 최근 2년 (730일) 검증')
print('=' * 60)
print(f'데이터: {df_2y.index[0]} ~ {df_2y.index[-1]}')
print(f'데이터 개수: {len(df_2y)}개')
print()

# RSI 계산
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df_2y['rsi'] = calculate_rsi(df_2y['Close'])
exp1 = df_2y['Close'].ewm(span=12, adjust=False).mean()
exp2 = df_2y['Close'].ewm(span=26, adjust=False).mean()
df_2y['MACD'] = exp1 - exp2

# ETH 파라미터
RSI_BUY, RSI_BUY_EXIT = 35, 40
RSI_SELL, RSI_SELL_EXIT = 85, 55

# 시그널
def find_signals(df):
    buy_signals, sell_signals = [], []
    in_oversold, in_overbought = False, False
    last_buy, last_sell = None, None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < RSI_BUY:
            in_oversold = True
            last_buy = df.index[idx]
        elif in_oversold and rsi >= RSI_BUY_EXIT and last_buy:
            buy_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_oversold = False
            last_buy = None
        
        if rsi > RSI_SELL:
            in_overbought = True
            last_sell = df.index[idx]
        elif in_overbought and rsi <= RSI_SELL_EXIT and last_sell:
            sell_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_overbought = False
            last_sell = None
    
    return buy_signals, sell_signals

buy_sigs, sell_sigs = find_signals(df_2y)
print(f'매수 시그널: {len(buy_sigs)}개')
print(f'매도 시그널: {len(sell_sigs)}개')
print()

# 시뮬레이션
CAPITAL = 1000
STOP_LOSS = -25
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE = 5
HEDGE_RATIO = 0.5
HEDGE_PROFIT = 5
HEDGE_STOP = -10

buy_dates = {s['date']: s for s in buy_sigs}
sell_dates = {s['date']: s for s in sell_sigs}

positions = []
long_trades = []
hedge_trades = []
current_hedge = None

for idx in range(len(df_2y)):
    date = df_2y.index[idx]
    price = df_2y['Close'].iloc[idx]
    high = df_2y['High'].iloc[idx]
    low = df_2y['Low'].iloc[idx]
    macd = df_2y['MACD'].iloc[idx]
    
    # 숏 청산
    if current_hedge:
        target = current_hedge['price'] * (1 - HEDGE_PROFIT / 100)
        stop = current_hedge['price'] * (1 - HEDGE_STOP / 100)
        
        if low <= target:
            profit = current_hedge['invested'] * HEDGE_PROFIT / 100
            hedge_trades.append({'profit': profit, 'invested': current_hedge['invested']})
            current_hedge = None
        elif high >= stop:
            profit = current_hedge['invested'] * HEDGE_STOP / 100
            hedge_trades.append({'profit': profit, 'invested': current_hedge['invested']})
            current_hedge = None
    
    # 롱 청산
    if positions:
        total_qty = sum(CAPITAL / p['price'] for p in positions)
        avg_price = (len(positions) * CAPITAL) / total_qty
        ret = (price / avg_price - 1) * 100
        
        exit_reason = None
        if ret <= STOP_LOSS:
            exit_reason = '손절'
            exit_price = avg_price * (1 + STOP_LOSS / 100)
        elif date in sell_dates:
            sell_ret = (sell_dates[date]['price'] / avg_price - 1) * 100
            if sell_ret > 0:
                exit_reason = '익절'
                exit_price = sell_dates[date]['price']
        
        if exit_reason:
            invested = len(positions) * CAPITAL
            final_ret = (exit_price / avg_price - 1) * 100
            profit = invested * final_ret / 100
            long_trades.append({'profit': profit, 'invested': invested, 'reason': exit_reason})
            
            if current_hedge:
                s_ret = (current_hedge['price'] - price) / current_hedge['price'] * 100
                s_profit = current_hedge['invested'] * s_ret / 100
                hedge_trades.append({'profit': s_profit, 'invested': current_hedge['invested']})
                current_hedge = None
            
            positions = []
    
    # 매수
    if date in buy_dates:
        positions.append({'date': date, 'price': buy_dates[date]['price']})
        num_buys = len(positions)
        
        should_hedge = False
        if num_buys == HEDGE_THRESHOLD and not current_hedge:
            should_hedge = True
        elif num_buys > HEDGE_THRESHOLD and HEDGE_UPGRADE > 0:
            if (num_buys - HEDGE_THRESHOLD) % HEDGE_UPGRADE == 0:
                should_hedge = True
        
        if should_hedge and macd < 0:
            if current_hedge:
                s_ret = (current_hedge['price'] - price) / current_hedge['price'] * 100
                s_profit = current_hedge['invested'] * s_ret / 100
                hedge_trades.append({'profit': s_profit, 'invested': current_hedge['invested']})
            
            current_hedge = {
                'price': price,
                'invested': num_buys * CAPITAL * HEDGE_RATIO
            }

# 결과
print('=' * 60)
print('OKX 5년 데이터 기준 최근 730일 결과')
print('=' * 60)

long_invested = sum(t['invested'] for t in long_trades)
long_profit = sum(t['profit'] for t in long_trades)
long_wins = sum(1 for t in long_trades if t['profit'] > 0)
long_stoploss = sum(1 for t in long_trades if t['reason'] == '손절')

short_invested = sum(t['invested'] for t in hedge_trades)
short_profit = sum(t['profit'] for t in hedge_trades)
short_wins = sum(1 for t in hedge_trades if t['profit'] > 0)

print(f'🟢 롱 성과')
print(f'   거래: {len(long_trades)}회')
print(f'   승률: {long_wins}/{len(long_trades)} ({long_wins/len(long_trades)*100:.0f}%)' if long_trades else '   승률: -')
print(f'   손절: {long_stoploss}회')
print(f'   투자금: ${long_invested:,}')
print(f'   손익: ${long_profit:+,.0f}')
print(f'   수익률: {long_profit/long_invested*100:+.2f}%' if long_invested > 0 else '')

print(f'\n🟣 숏 헷징 성과')
print(f'   거래: {len(hedge_trades)}회')
print(f'   승률: {short_wins}/{len(hedge_trades)} ({short_wins/len(hedge_trades)*100:.0f}%)' if hedge_trades else '   승률: -')
print(f'   투자금: ${short_invested:,.0f}')
print(f'   손익: ${short_profit:+,.0f}')
print(f'   수익률: {short_profit/short_invested*100:+.2f}%' if short_invested > 0 else '')

print(f'\n💰 총 성과')
total_invested = long_invested + short_invested
total_profit = long_profit + short_profit
print(f'   총 투자금: ${total_invested:,.0f}')
print(f'   총 손익: ${total_profit:+,.0f}')
print(f'   수익률: {total_profit/total_invested*100:+.2f}%' if total_invested > 0 else '')

print('\n' + '=' * 60)
print('대시보드 결과와 비교')
print('=' * 60)
print('대시보드 (yfinance):')
print('   롱: 19회, +$1,704')
print('   숏: 26회, -$378')
print('   총: +$1,326')
print()
print('OKX 5년 데이터:')
print(f'   롱: {len(long_trades)}회, ${long_profit:+,.0f}')
print(f'   숏: {len(hedge_trades)}회, ${short_profit:+,.0f}')
print(f'   총: ${total_profit:+,.0f}')

