"""
ETH 파라미터 최적화
- 577개 조합 테스트 (576개 헷징 + 1개 롱온리)
- BTC 대시보드와 동일한 로직 사용
"""
import pandas as pd
import numpy as np
from itertools import product
import time

# 데이터 로드
df = pd.read_csv('data/eth_4h_5y.csv', index_col='Date', parse_dates=True)
print('=' * 70)
print('ETH 파라미터 최적화 (577개 조합)')
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
df['MA40'] = df['Close'].rolling(window=40).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['golden_cross'] = df['MA40'] > df['MA200']

# MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2

# 고정 파라미터 (BTC 대시보드 기본값)
RSI_BUY = 35
RSI_BUY_EXIT = 40
RSI_SELL = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False
CAPITAL_PER_ENTRY = 1000

# 시그널 생성 (한번만)
def find_buy_signals(df):
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < RSI_BUY:
            in_oversold = True
            last_signal_date = df.index[idx]
        else:
            if in_oversold and rsi >= RSI_BUY_EXIT and last_signal_date is not None:
                buy_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_oversold = False
                last_signal_date = None
    
    return buy_signals

def find_sell_signals(df):
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > RSI_SELL:
            in_overbought = True
            last_signal_date = df.index[idx]
        else:
            if in_overbought and rsi <= RSI_SELL_EXIT and last_signal_date is not None:
                sell_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals

# 시뮬레이션 함수
def simulate_trades(df, buy_signals, sell_signals, use_hedge=False,
                   hedge_threshold=2, hedge_upgrade_interval=3,
                   hedge_ratio=1.0, hedge_profit=8, hedge_stop=-15):
    
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    positions = []
    trades = []
    current_hedge = None
    hedge_trades = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        current_high = df['High'].iloc[idx]
        current_low = df['Low'].iloc[idx]
        macd_val = df['MACD'].iloc[idx]
        
        # 숏 헷징 청산 체크
        if use_hedge and current_hedge is not None:
            target_price = current_hedge['entry_price'] * (1 - hedge_profit / 100)
            stop_price_hedge = current_hedge['entry_price'] * (1 - hedge_stop / 100)
            
            short_exit_reason = None
            short_exit_price = None
            
            if current_low <= target_price:
                short_exit_reason = "익절"
                short_exit_price = target_price
            elif current_high >= stop_price_hedge:
                short_exit_reason = "손절"
                short_exit_price = stop_price_hedge
            
            if short_exit_reason:
                short_return = (current_hedge['entry_price'] - short_exit_price) / current_hedge['entry_price'] * 100
                short_profit = current_hedge['invested'] * short_return / 100
                hedge_trades.append({
                    'invested': current_hedge['invested'],
                    'profit': short_profit
                })
                current_hedge = None
        
        # 롱 포지션 처리
        if positions:
            total_qty = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = (len(positions) * CAPITAL_PER_ENTRY) / total_qty
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = None
            
            if current_return <= STOP_LOSS:
                exit_reason = "손절"
                exit_price = avg_price * (1 + STOP_LOSS / 100)
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                total_invested = len(positions) * CAPITAL_PER_ENTRY
                final_return = (exit_price / avg_price - 1) * 100
                profit = total_invested * final_return / 100
                
                trades.append({
                    'invested': total_invested,
                    'profit': profit
                })
                
                # 롱 청산시 숏도 청산
                if use_hedge and current_hedge is not None:
                    short_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
                    short_profit = current_hedge['invested'] * short_return / 100
                    hedge_trades.append({
                        'invested': current_hedge['invested'],
                        'profit': short_profit
                    })
                    current_hedge = None
                
                positions = []
        
        # 매수 처리
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
            
            num_buys = len(positions)
            
            # 헷징 진입/업그레이드 체크
            if use_hedge:
                should_hedge = False
                if num_buys == hedge_threshold and current_hedge is None:
                    should_hedge = True
                elif num_buys > hedge_threshold and hedge_upgrade_interval > 0:
                    if (num_buys - hedge_threshold) % hedge_upgrade_interval == 0:
                        should_hedge = True
                
                if should_hedge and macd_val < 0:
                    # 기존 숏 청산 (업그레이드)
                    if current_hedge is not None:
                        short_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
                        short_profit = current_hedge['invested'] * short_return / 100
                        hedge_trades.append({
                            'invested': current_hedge['invested'],
                            'profit': short_profit
                        })
                    
                    # 새 숏 진입
                    current_hedge = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'invested': num_buys * CAPITAL_PER_ENTRY * hedge_ratio
                    }
    
    return trades, hedge_trades

# 시그널 미리 계산
print('시그널 계산 중...')
buy_signals = find_buy_signals(df)
sell_signals = find_sell_signals(df)
print(f'매수 시그널: {len(buy_signals)}개')
print(f'매도 시그널: {len(sell_signals)}개')
print()

# 최적화 파라미터 범위
hedge_thresholds = [2, 3, 4]
hedge_upgrade_intervals = [0, 3, 5]
hedge_ratios = [0.5, 0.75, 1.0, 1.25]
hedge_profits = [5, 8, 10, 12]
hedge_stops = [-10, -12, -15, -20]

# 결과 저장
results = []

# 1. 롱 온리 테스트
print('테스트 중...')
start_time = time.time()

trades, _ = simulate_trades(df, buy_signals, sell_signals, use_hedge=False)
long_invested = sum(t['invested'] for t in trades)
long_profit = sum(t['profit'] for t in trades)
long_return = long_profit / long_invested * 100 if long_invested > 0 else 0

results.append({
    'type': '롱 온리',
    'hedge_threshold': '-',
    'hedge_upgrade': '-',
    'hedge_ratio': '-',
    'hedge_profit': '-',
    'hedge_stop': '-',
    'long_trades': len(trades),
    'long_invested': long_invested,
    'long_profit': long_profit,
    'long_return': long_return,
    'short_trades': 0,
    'short_invested': 0,
    'short_profit': 0,
    'short_return': 0,
    'total_profit': long_profit,
    'total_return': long_return
})

# 2. 헷징 조합 테스트
total_combos = len(hedge_thresholds) * len(hedge_upgrade_intervals) * len(hedge_ratios) * len(hedge_profits) * len(hedge_stops)
combo_count = 0

for ht, hui, hr, hp, hs in product(hedge_thresholds, hedge_upgrade_intervals, 
                                    hedge_ratios, hedge_profits, hedge_stops):
    combo_count += 1
    
    if combo_count % 100 == 0:
        print(f'  진행: {combo_count}/{total_combos} ({combo_count/total_combos*100:.0f}%)')
    
    trades, hedge_trades = simulate_trades(
        df, buy_signals, sell_signals,
        use_hedge=True,
        hedge_threshold=ht,
        hedge_upgrade_interval=hui,
        hedge_ratio=hr,
        hedge_profit=hp,
        hedge_stop=hs
    )
    
    long_invested = sum(t['invested'] for t in trades)
    long_profit = sum(t['profit'] for t in trades)
    long_return = long_profit / long_invested * 100 if long_invested > 0 else 0
    
    short_invested = sum(h['invested'] for h in hedge_trades)
    short_profit = sum(h['profit'] for h in hedge_trades)
    short_return = short_profit / short_invested * 100 if short_invested > 0 else 0
    
    total_invested = long_invested + short_invested
    total_profit = long_profit + short_profit
    total_return = total_profit / total_invested * 100 if total_invested > 0 else 0
    
    results.append({
        'type': '헷징',
        'hedge_threshold': ht,
        'hedge_upgrade': hui,
        'hedge_ratio': hr,
        'hedge_profit': hp,
        'hedge_stop': hs,
        'long_trades': len(trades),
        'long_invested': long_invested,
        'long_profit': long_profit,
        'long_return': long_return,
        'short_trades': len(hedge_trades),
        'short_invested': short_invested,
        'short_profit': short_profit,
        'short_return': short_return,
        'total_profit': total_profit,
        'total_return': total_return
    })

elapsed = time.time() - start_time
print(f'\n완료! ({elapsed:.1f}초)')

# 결과 정렬 (총 수익률 기준)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('total_return', ascending=False)

# Top 10 출력
print('\n' + '=' * 70)
print('📊 ETH 최적화 결과 - Top 10 (총 수익률 기준)')
print('=' * 70)

for i, row in results_df.head(10).iterrows():
    rank = results_df.index.get_loc(i) + 1
    print(f'\n{rank}위: {row["type"]}')
    if row['type'] == '헷징':
        print(f'   파라미터: threshold={row["hedge_threshold"]}, upgrade={row["hedge_upgrade"]}, '
              f'ratio={row["hedge_ratio"]}, profit={row["hedge_profit"]}%, stop={row["hedge_stop"]}%')
    print(f'   롱: {row["long_trades"]}회, ${row["long_profit"]:+,.0f} ({row["long_return"]:+.2f}%)')
    print(f'   숏: {row["short_trades"]}회, ${row["short_profit"]:+,.0f} ({row["short_return"]:+.2f}%)')
    print(f'   💰 총: ${row["total_profit"]:+,.0f} ({row["total_return"]:+.2f}%)')

# 롱 온리 결과 별도 출력
long_only = results_df[results_df['type'] == '롱 온리'].iloc[0]
long_only_rank = results_df.index.get_loc(results_df[results_df['type'] == '롱 온리'].index[0]) + 1

print('\n' + '=' * 70)
print(f'📌 롱 온리 결과 (순위: {long_only_rank}위 / {len(results_df)}개)')
print('=' * 70)
print(f'   롱: {long_only["long_trades"]}회, ${long_only["long_profit"]:+,.0f} ({long_only["long_return"]:+.2f}%)')

# 최고 헷징 vs 롱 온리 비교
best_hedge = results_df[results_df['type'] == '헷징'].iloc[0]
print('\n' + '=' * 70)
print('📈 비교: 최고 헷징 vs 롱 온리')
print('=' * 70)
print(f'롱 온리:    ${long_only["total_profit"]:+,.0f} ({long_only["total_return"]:+.2f}%)')
print(f'최고 헷징:  ${best_hedge["total_profit"]:+,.0f} ({best_hedge["total_return"]:+.2f}%)')

diff = best_hedge["total_profit"] - long_only["total_profit"]
if diff > 0:
    print(f'→ 헷징이 ${diff:+,.0f} 더 좋음! ✅')
else:
    print(f'→ 롱 온리가 ${-diff:+,.0f} 더 좋음! 🎯')

# 숏 수익 양수인 헷징 조합 수
profitable_hedges = results_df[(results_df['type'] == '헷징') & (results_df['short_profit'] > 0)]
print(f'\n숏 수익 양수인 조합: {len(profitable_hedges)}개 / 576개')

# CSV 저장
results_df.to_csv('data/eth_optimization_results.csv', index=False)
print(f'\n결과 저장: data/eth_optimization_results.csv')

