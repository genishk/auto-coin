"""
ETH 롱 파라미터 최적화 - 상위 3개 연도별 분석
"""
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('data/eth_4h_5y.csv', index_col='Date', parse_dates=True)

# 기술 지표
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
HEDGE_THRESHOLD = 2
HEDGE_UPGRADE = 5
HEDGE_RATIO = 0.5
HEDGE_PROFIT = 5
HEDGE_STOP = -10

# 테스트할 파라미터 세트
param_sets = [
    {'name': '1위 최적', 'rsi_buy': 35, 'rsi_buy_exit': 40, 'rsi_sell': 85, 'rsi_sell_exit': 55},
    {'name': '3위', 'rsi_buy': 35, 'rsi_buy_exit': 40, 'rsi_sell': 85, 'rsi_sell_exit': 50},
    {'name': 'BTC 기본', 'rsi_buy': 35, 'rsi_buy_exit': 40, 'rsi_sell': 80, 'rsi_sell_exit': 55},
]

def find_signals(df_year, rsi_buy, rsi_buy_exit, rsi_sell, rsi_sell_exit):
    buy_signals, sell_signals = [], []
    in_oversold, in_overbought = False, False
    last_buy, last_sell = None, None
    
    for idx in range(len(df_year)):
        rsi = df_year['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_buy:
            in_oversold = True
            last_buy = df_year.index[idx]
        elif in_oversold and rsi >= rsi_buy_exit and last_buy:
            buy_signals.append({'date': df_year.index[idx], 'price': df_year['Close'].iloc[idx]})
            in_oversold = False
            last_buy = None
        
        if rsi > rsi_sell:
            in_overbought = True
            last_sell = df_year.index[idx]
        elif in_overbought and rsi <= rsi_sell_exit and last_sell:
            sell_signals.append({'date': df_year.index[idx], 'price': df_year['Close'].iloc[idx]})
            in_overbought = False
            last_sell = None
    
    return buy_signals, sell_signals

def simulate(df_year, buy_signals, sell_signals, use_hedge=True):
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
        
        if use_hedge and current_hedge:
            target = current_hedge['price'] * (1 - HEDGE_PROFIT / 100)
            stop = current_hedge['price'] * (1 - HEDGE_STOP / 100)
            
            if low <= target:
                hedge_trades.append({'profit': current_hedge['invested'] * HEDGE_PROFIT / 100})
                current_hedge = None
            elif high >= stop:
                hedge_trades.append({'profit': current_hedge['invested'] * HEDGE_STOP / 100})
                current_hedge = None
        
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
                long_trades.append({'profit': invested * final_ret / 100, 'reason': exit_reason})
                
                if use_hedge and current_hedge:
                    s_ret = (current_hedge['price'] - price) / current_hedge['price'] * 100
                    hedge_trades.append({'profit': current_hedge['invested'] * s_ret / 100})
                    current_hedge = None
                
                positions = []
        
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
                        hedge_trades.append({'profit': current_hedge['invested'] * s_ret / 100})
                    
                    current_hedge = {
                        'price': price,
                        'invested': num_buys * CAPITAL * HEDGE_RATIO
                    }
    
    return long_trades, hedge_trades

years = [2020, 2021, 2022, 2023, 2024, 2025]

print('=' * 90)
print('ETH 롱 파라미터 최적화 - 연도별 성과 비교')
print('=' * 90)

# 각 파라미터 세트별 연도별 결과
all_results = {}

for params in param_sets:
    name = params['name']
    all_results[name] = {'yearly': [], 'total_long': 0, 'total_short': 0, 'total': 0}
    
    for year in years:
        df_year = df[(df.index >= f'{year}-01-01') & (df.index <= f'{year}-12-31')]
        if len(df_year) == 0:
            continue
        
        # ETH 변동
        eth_change = (df_year['Close'].iloc[-1] / df_year['Close'].iloc[0] - 1) * 100
        
        buy_sigs, sell_sigs = find_signals(df_year, params['rsi_buy'], params['rsi_buy_exit'],
                                           params['rsi_sell'], params['rsi_sell_exit'])
        
        long_trades, hedge_trades = simulate(df_year, buy_sigs, sell_sigs, use_hedge=True)
        
        long_profit = sum(t['profit'] for t in long_trades)
        short_profit = sum(t['profit'] for t in hedge_trades)
        stoploss = sum(1 for t in long_trades if t['reason'] == '손절')
        
        all_results[name]['yearly'].append({
            'year': year,
            'eth': eth_change,
            'trades': len(long_trades),
            'stoploss': stoploss,
            'long': long_profit,
            'short': short_profit,
            'total': long_profit + short_profit
        })
        
        all_results[name]['total_long'] += long_profit
        all_results[name]['total_short'] += short_profit
        all_results[name]['total'] += long_profit + short_profit

# 결과 출력
for params in param_sets:
    name = params['name']
    results = all_results[name]
    
    print(f'\n{"=" * 90}')
    print(f'📊 {name}: RSI {params["rsi_buy"]}/{params["rsi_buy_exit"]} → {params["rsi_sell"]}/{params["rsi_sell_exit"]}')
    print('=' * 90)
    
    print(f'{"연도":^6} | {"ETH":^8} | {"거래":^10} | {"롱 수익":^12} | {"숏 수익":^12} | {"총 수익":^12}')
    print('-' * 70)
    
    for r in results['yearly']:
        sl_str = f'(손절{r["stoploss"]})' if r['stoploss'] > 0 else ''
        print(f'{r["year"]:^6} | {r["eth"]:+6.0f}% | {r["trades"]:>3}회 {sl_str:^5} | '
              f'${r["long"]:+8,.0f} | ${r["short"]:+8,.0f} | ${r["total"]:+8,.0f}')
    
    print('-' * 70)
    print(f'{"합계":^6} | {"":^8} | {"":^10} | '
          f'${results["total_long"]:+8,.0f} | ${results["total_short"]:+8,.0f} | ${results["total"]:+8,.0f}')

# 최종 비교
print('\n' + '=' * 90)
print('📈 최종 비교')
print('=' * 90)

for params in param_sets:
    name = params['name']
    results = all_results[name]
    print(f'{name:12}: 롱 ${results["total_long"]:+,.0f} + 숏 ${results["total_short"]:+,.0f} = 총 ${results["total"]:+,.0f}')

best_name = max(all_results.keys(), key=lambda x: all_results[x]['total'])
print(f'\n🏆 최고: {best_name} (${all_results[best_name]["total"]:+,.0f})')

# 하락장(2022) 비교
print('\n' + '=' * 90)
print('🔴 하락장(2022) 성과 비교')
print('=' * 90)

for params in param_sets:
    name = params['name']
    y2022 = [r for r in all_results[name]['yearly'] if r['year'] == 2022][0]
    print(f'{name:12}: 롱 ${y2022["long"]:+,.0f} + 숏 ${y2022["short"]:+,.0f} = ${y2022["total"]:+,.0f}')

