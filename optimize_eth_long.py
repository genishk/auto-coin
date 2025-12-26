"""
ETH 롱 파라미터 최적화
- 108개 조합 테스트
- 헷징은 최적값 고정 (threshold=2, upgrade=5, ratio=50%, profit=5%, stop=-10%)
"""
import pandas as pd
import numpy as np
from itertools import product
import time

# 데이터 로드
df = pd.read_csv('data/eth_4h_5y.csv', index_col='Date', parse_dates=True)
print('=' * 70)
print('ETH 롱 파라미터 최적화 (108개 조합)')
print('=' * 70)
print(f'데이터: {df.index[0].strftime("%Y-%m-%d")} ~ {df.index[-1].strftime("%Y-%m-%d")} ({len(df)}개)')
print()

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

# 고정 파라미터
STOP_LOSS = -25
CAPITAL = 1000

# 헷징 파라미터 (최적값 고정)
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE = 5
HEDGE_RATIO = 0.5
HEDGE_PROFIT = 5
HEDGE_STOP = -10

def find_buy_signals(df, rsi_buy, rsi_buy_exit):
    buy_signals = []
    in_oversold = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_buy:
            in_oversold = True
            last_date = df.index[idx]
        elif in_oversold and rsi >= rsi_buy_exit and last_date:
            buy_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_oversold = False
            last_date = None
    
    return buy_signals

def find_sell_signals(df, rsi_sell, rsi_sell_exit):
    sell_signals = []
    in_overbought = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_sell:
            in_overbought = True
            last_date = df.index[idx]
        elif in_overbought and rsi <= rsi_sell_exit and last_date:
            sell_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
            in_overbought = False
            last_date = None
    
    return sell_signals

def simulate_trades(df, buy_signals, sell_signals, use_hedge=True):
    buy_dates = {s['date']: s for s in buy_signals}
    sell_dates = {s['date']: s for s in sell_signals}
    
    positions = []
    long_trades = []
    hedge_trades = []
    current_hedge = None
    
    for idx in range(len(df)):
        date = df.index[idx]
        price = df['Close'].iloc[idx]
        high = df['High'].iloc[idx]
        low = df['Low'].iloc[idx]
        macd = df['MACD'].iloc[idx]
        
        # 숏 청산
        if use_hedge and current_hedge:
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

# 테스트 파라미터
rsi_buys = [30, 35, 40]
rsi_buy_exits = [35, 40, 45, 50]
rsi_sells = [75, 80, 85]
rsi_sell_exits = [50, 55, 60]

results = []
total = len(rsi_buys) * len(rsi_buy_exits) * len(rsi_sells) * len(rsi_sell_exits)

print(f'테스트 중... (총 {total}개 조합)')
start_time = time.time()

count = 0
for rb, rbe, rs, rse in product(rsi_buys, rsi_buy_exits, rsi_sells, rsi_sell_exits):
    count += 1
    
    # 유효성 체크: 매수 탈출 > 매수 진입, 매도 탈출 < 매도 진입
    if rbe <= rb or rse >= rs:
        continue
    
    buy_signals = find_buy_signals(df, rb, rbe)
    sell_signals = find_sell_signals(df, rs, rse)
    
    # 롱 온리
    long_only, _ = simulate_trades(df, buy_signals, sell_signals, use_hedge=False)
    long_only_invested = sum(t['invested'] for t in long_only)
    long_only_profit = sum(t['profit'] for t in long_only)
    long_only_stoploss = sum(1 for t in long_only if t['reason'] == '손절')
    
    # 헷징
    long_hedge, hedge = simulate_trades(df, buy_signals, sell_signals, use_hedge=True)
    long_hedge_invested = sum(t['invested'] for t in long_hedge)
    long_hedge_profit = sum(t['profit'] for t in long_hedge)
    short_invested = sum(t['invested'] for t in hedge)
    short_profit = sum(t['profit'] for t in hedge)
    
    total_invested = long_hedge_invested + short_invested
    total_profit = long_hedge_profit + short_profit
    total_return = total_profit / total_invested * 100 if total_invested > 0 else 0
    
    results.append({
        'rsi_buy': rb,
        'rsi_buy_exit': rbe,
        'rsi_sell': rs,
        'rsi_sell_exit': rse,
        'long_trades': len(long_only),
        'long_stoploss': long_only_stoploss,
        'long_only_profit': long_only_profit,
        'long_only_return': long_only_profit / long_only_invested * 100 if long_only_invested > 0 else 0,
        'hedge_long_profit': long_hedge_profit,
        'hedge_short_profit': short_profit,
        'hedge_total_profit': total_profit,
        'hedge_total_return': total_return
    })

elapsed = time.time() - start_time
print(f'완료! ({elapsed:.1f}초, 유효 조합: {len(results)}개)')

# 결과 정렬
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('hedge_total_return', ascending=False)

# Top 10 출력
print('\n' + '=' * 70)
print('📊 ETH 롱 파라미터 최적화 결과 - Top 10 (헷징 포함 수익률 기준)')
print('=' * 70)

for i, (idx, row) in enumerate(results_df.head(10).iterrows()):
    print(f'\n{i+1}위: RSI {row["rsi_buy"]}/{row["rsi_buy_exit"]} → {row["rsi_sell"]}/{row["rsi_sell_exit"]}')
    print(f'   거래: {row["long_trades"]}회 (손절 {row["long_stoploss"]}회)')
    print(f'   롱 온리: ${row["long_only_profit"]:+,.0f} ({row["long_only_return"]:+.2f}%)')
    print(f'   헷징: 롱 ${row["hedge_long_profit"]:+,.0f} + 숏 ${row["hedge_short_profit"]:+,.0f} = ${row["hedge_total_profit"]:+,.0f} ({row["hedge_total_return"]:+.2f}%)')

# 현재 BTC 파라미터와 비교
btc_params = results_df[(results_df['rsi_buy'] == 35) & (results_df['rsi_buy_exit'] == 40) & 
                        (results_df['rsi_sell'] == 80) & (results_df['rsi_sell_exit'] == 55)]

print('\n' + '=' * 70)
print('📌 현재 BTC 파라미터 (35/40 → 80/55) 결과')
print('=' * 70)

if len(btc_params) > 0:
    row = btc_params.iloc[0]
    rank = results_df.index.get_loc(btc_params.index[0]) + 1
    print(f'순위: {rank}위 / {len(results_df)}개')
    print(f'거래: {row["long_trades"]}회 (손절 {row["long_stoploss"]}회)')
    print(f'롱 온리: ${row["long_only_profit"]:+,.0f} ({row["long_only_return"]:+.2f}%)')
    print(f'헷징: ${row["hedge_total_profit"]:+,.0f} ({row["hedge_total_return"]:+.2f}%)')

# 최고 vs BTC 비교
print('\n' + '=' * 70)
print('📈 최고 파라미터 vs BTC 파라미터')
print('=' * 70)

best = results_df.iloc[0]
print(f'최고: RSI {best["rsi_buy"]}/{best["rsi_buy_exit"]} → {best["rsi_sell"]}/{best["rsi_sell_exit"]}')
print(f'      헷징 수익: ${best["hedge_total_profit"]:+,.0f} ({best["hedge_total_return"]:+.2f}%)')

if len(btc_params) > 0:
    btc = btc_params.iloc[0]
    print(f'BTC:  RSI 35/40 → 80/55')
    print(f'      헷징 수익: ${btc["hedge_total_profit"]:+,.0f} ({btc["hedge_total_return"]:+.2f}%)')
    
    diff = best["hedge_total_profit"] - btc["hedge_total_profit"]
    print(f'\n→ 최적화로 ${diff:+,.0f} 추가 수익 가능!')

# CSV 저장
results_df.to_csv('data/eth_long_optimization_results.csv', index=False)
print(f'\n결과 저장: data/eth_long_optimization_results.csv')

