"""
ETH 연도별 성과 분석
- 롱 온리 vs 최고 헷징 비교
"""
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data/eth_4h_5y.csv', index_col='Date', parse_dates=True)

# 기술 지표 계산
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['Close'])
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2

# 파라미터
RSI_BUY, RSI_BUY_EXIT = 35, 40
RSI_SELL, RSI_SELL_EXIT = 80, 55
STOP_LOSS = -25
CAPITAL = 1000

# 최고 헷징 파라미터 (3위: upgrade=5가 숏 수익 양수)
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE = 5
HEDGE_RATIO = 0.5
HEDGE_PROFIT = 5
HEDGE_STOP = -10

def find_signals(df):
    buy_signals, sell_signals = [], []
    in_oversold, in_overbought = False, False
    last_buy_date, last_sell_date = None, None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        # 매수
        if rsi < RSI_BUY:
            in_oversold = True
            last_buy_date = df.index[idx]
        elif in_oversold and rsi >= RSI_BUY_EXIT and last_buy_date:
            buy_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_oversold = False
            last_buy_date = None
        
        # 매도
        if rsi > RSI_SELL:
            in_overbought = True
            last_sell_date = df.index[idx]
        elif in_overbought and rsi <= RSI_SELL_EXIT and last_sell_date:
            sell_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_overbought = False
            last_sell_date = None
    
    return buy_signals, sell_signals

def simulate_year(df_year, use_hedge=False):
    buy_signals, sell_signals = find_signals(df_year)
    buy_dates = {s['date']: s for s in buy_signals}
    sell_dates = {s['date']: s for s in sell_signals}
    
    positions = []
    long_trades = []
    hedge_trades = []
    current_hedge = None
    
    for idx in range(len(df_year)):
        date = df_year.index[idx]
        price = df_year['Close'].iloc[idx]
        high = df_year['High'].iloc[idx]
        low = df_year['Low'].iloc[idx]
        macd = df_year['MACD'].iloc[idx]
        
        # 숏 청산 체크
        if use_hedge and current_hedge:
            target = current_hedge['price'] * (1 - HEDGE_PROFIT / 100)
            stop = current_hedge['price'] * (1 - HEDGE_STOP / 100)
            
            if low <= target:
                ret = HEDGE_PROFIT
                profit = current_hedge['invested'] * ret / 100
                hedge_trades.append({'profit': profit, 'invested': current_hedge['invested']})
                current_hedge = None
            elif high >= stop:
                ret = HEDGE_STOP
                profit = current_hedge['invested'] * ret / 100
                hedge_trades.append({'profit': profit, 'invested': current_hedge['invested']})
                current_hedge = None
        
        # 롱 청산 체크
        if positions:
            total_qty = sum(CAPITAL / p['price'] for p in positions)
            avg_price = (len(positions) * CAPITAL) / total_qty
            ret = (price / avg_price - 1) * 100
            
            exit_reason = None
            if ret <= STOP_LOSS:
                exit_reason = "손절"
                exit_price = avg_price * (1 + STOP_LOSS / 100)
            elif date in sell_dates:
                sell_ret = (sell_dates[date]['price'] / avg_price - 1) * 100
                if sell_ret > 0:
                    exit_reason = "익절"
                    exit_price = sell_dates[date]['price']
            
            if exit_reason:
                invested = len(positions) * CAPITAL
                final_ret = (exit_price / avg_price - 1) * 100
                profit = invested * final_ret / 100
                long_trades.append({'profit': profit, 'invested': invested, 'reason': exit_reason})
                
                # 숏도 청산
                if use_hedge and current_hedge:
                    s_ret = (current_hedge['price'] - price) / current_hedge['price'] * 100
                    s_profit = current_hedge['invested'] * s_ret / 100
                    hedge_trades.append({'profit': s_profit, 'invested': current_hedge['invested']})
                    current_hedge = None
                
                positions = []
        
        # 매수
        if date in buy_dates:
            positions.append({'date': date, 'price': buy_dates[date]['price']})
            num_buys = len(positions)
            
            if use_hedge:
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
    
    return long_trades, hedge_trades

# 연도별 분석
years = [2020, 2021, 2022, 2023, 2024, 2025]

print('=' * 80)
print('ETH 연도별 성과 분석: 롱 온리 vs 최고 헷징')
print('=' * 80)
print(f'헷징 파라미터: threshold={HEDGE_THRESHOLD}, upgrade={HEDGE_UPGRADE}, '
      f'ratio={HEDGE_RATIO*100:.0f}%, profit={HEDGE_PROFIT}%, stop={HEDGE_STOP}%')
print()

results = []

for year in years:
    # 연도 데이터 필터
    year_start = f'{year}-01-01'
    year_end = f'{year}-12-31'
    df_year = df[(df.index >= year_start) & (df.index <= year_end)]
    
    if len(df_year) == 0:
        continue
    
    # ETH 가격 변화
    start_price = df_year['Close'].iloc[0]
    end_price = df_year['Close'].iloc[-1]
    eth_change = (end_price / start_price - 1) * 100
    
    # 롱 온리
    long_trades_only, _ = simulate_year(df_year, use_hedge=False)
    long_only_invested = sum(t['invested'] for t in long_trades_only)
    long_only_profit = sum(t['profit'] for t in long_trades_only)
    long_only_stoploss = sum(1 for t in long_trades_only if t['reason'] == '손절')
    
    # 헷징
    long_trades_hedge, hedge_trades = simulate_year(df_year, use_hedge=True)
    long_hedge_invested = sum(t['invested'] for t in long_trades_hedge)
    long_hedge_profit = sum(t['profit'] for t in long_trades_hedge)
    short_invested = sum(t['invested'] for t in hedge_trades)
    short_profit = sum(t['profit'] for t in hedge_trades)
    
    total_invested = long_hedge_invested + short_invested
    total_profit = long_hedge_profit + short_profit
    
    results.append({
        'year': year,
        'eth_change': eth_change,
        'long_only_trades': len(long_trades_only),
        'long_only_invested': long_only_invested,
        'long_only_profit': long_only_profit,
        'long_only_stoploss': long_only_stoploss,
        'hedge_long_trades': len(long_trades_hedge),
        'hedge_long_profit': long_hedge_profit,
        'hedge_short_trades': len(hedge_trades),
        'hedge_short_profit': short_profit,
        'hedge_total_profit': total_profit
    })

# 결과 출력
print(f'{"연도":^6} | {"ETH":^8} | {"롱온리":^20} | {"헷징 롱":^12} | {"헷징 숏":^12} | {"헷징 총":^12}')
print(f'{"":-^6} | {"":-^8} | {"":-^20} | {"":-^12} | {"":-^12} | {"":-^12}')

for r in results:
    long_only_str = f'{r["long_only_trades"]}회 ${r["long_only_profit"]:+,.0f}'
    if r['long_only_stoploss'] > 0:
        long_only_str += f' (손절{r["long_only_stoploss"]})'
    
    print(f'{r["year"]:^6} | {r["eth_change"]:+6.0f}% | {long_only_str:^20} | '
          f'${r["hedge_long_profit"]:+6,.0f} | ${r["hedge_short_profit"]:+6,.0f} | ${r["hedge_total_profit"]:+6,.0f}')

# 합계
print(f'{"":-^6} | {"":-^8} | {"":-^20} | {"":-^12} | {"":-^12} | {"":-^12}')
total_long_only = sum(r['long_only_profit'] for r in results)
total_hedge_long = sum(r['hedge_long_profit'] for r in results)
total_hedge_short = sum(r['hedge_short_profit'] for r in results)
total_hedge_total = sum(r['hedge_total_profit'] for r in results)
print(f'{"합계":^6} | {"":^8} | ${total_long_only:+,.0f} {"":^10} | '
      f'${total_hedge_long:+6,.0f} | ${total_hedge_short:+6,.0f} | ${total_hedge_total:+6,.0f}')

print()
print('=' * 80)
print('📊 연도별 상세 분석')
print('=' * 80)

for r in results:
    year = r['year']
    eth_change = r['eth_change']
    
    # 시장 상태 판단
    if eth_change > 50:
        market = "🟢 강한 상승장"
    elif eth_change > 0:
        market = "🟡 약한 상승장"
    elif eth_change > -30:
        market = "🟠 약한 하락장"
    else:
        market = "🔴 강한 하락장"
    
    print(f'\n{year}년: {market} (ETH {eth_change:+.0f}%)')
    print(f'  롱 온리: ${r["long_only_profit"]:+,.0f} ({r["long_only_trades"]}회, 손절 {r["long_only_stoploss"]}회)')
    print(f'  헷징:    롱 ${r["hedge_long_profit"]:+,.0f} + 숏 ${r["hedge_short_profit"]:+,.0f} = ${r["hedge_total_profit"]:+,.0f}')
    
    diff = r['hedge_total_profit'] - r['long_only_profit']
    if diff > 0:
        print(f'  → 헷징이 ${diff:+,.0f} 더 좋음')
    else:
        print(f'  → 롱 온리가 ${-diff:+,.0f} 더 좋음')

print()
print('=' * 80)
print('📈 최종 비교')
print('=' * 80)
print(f'롱 온리 총 수익: ${total_long_only:+,.0f}')
print(f'헷징 총 수익:    ${total_hedge_total:+,.0f} (롱 ${total_hedge_long:+,.0f} + 숏 ${total_hedge_short:+,.0f})')

diff = total_hedge_total - total_long_only
if diff > 0:
    print(f'\n→ 헷징이 ${diff:+,.0f} 더 좋음!')
else:
    print(f'\n→ 롱 온리가 ${-diff:+,.0f} 더 좋음!')

