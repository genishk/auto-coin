"""
ETH 최종 전략 상세 성과 분석
- RSI 35/40 → 85/55
- 헷징: threshold=2, upgrade=5, ratio=50%, profit=5%, stop=-10%
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# 최종 파라미터
RSI_BUY = 35
RSI_BUY_EXIT = 40
RSI_SELL = 85
RSI_SELL_EXIT = 55
STOP_LOSS = -25
CAPITAL = 1000

HEDGE_THRESHOLD = 2
HEDGE_UPGRADE = 5
HEDGE_RATIO = 0.5
HEDGE_PROFIT = 5
HEDGE_STOP = -10

def find_signals(df_period):
    buy_signals, sell_signals = [], []
    in_oversold, in_overbought = False, False
    last_buy, last_sell = None, None
    
    for idx in range(len(df_period)):
        rsi = df_period['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < RSI_BUY:
            in_oversold = True
            last_buy = df_period.index[idx]
        elif in_oversold and rsi >= RSI_BUY_EXIT and last_buy:
            buy_signals.append({'date': df_period.index[idx], 'price': df_period['Close'].iloc[idx]})
            in_oversold = False
            last_buy = None
        
        if rsi > RSI_SELL:
            in_overbought = True
            last_sell = df_period.index[idx]
        elif in_overbought and rsi <= RSI_SELL_EXIT and last_sell:
            sell_signals.append({'date': df_period.index[idx], 'price': df_period['Close'].iloc[idx]})
            in_overbought = False
            last_sell = None
    
    return buy_signals, sell_signals

def simulate_detailed(df_period, buy_signals, sell_signals):
    buy_dates = {s['date']: s for s in buy_signals}
    sell_dates = {s['date']: s for s in sell_signals}
    
    positions = []
    long_trades = []
    hedge_trades = []
    current_hedge = None
    
    max_water = 0  # 최대 물타기
    water_counts = []  # 각 거래별 물타기 횟수
    
    for idx in range(len(df_period)):
        date = df_period.index[idx]
        price = df_period['Close'].iloc[idx]
        high = df_period['High'].iloc[idx]
        low = df_period['Low'].iloc[idx]
        macd = df_period['MACD'].iloc[idx]
        
        # 숏 청산
        if current_hedge:
            target = current_hedge['price'] * (1 - HEDGE_PROFIT / 100)
            stop = current_hedge['price'] * (1 - HEDGE_STOP / 100)
            
            exit_reason = None
            exit_price = None
            if low <= target:
                exit_reason = "익절"
                exit_price = target
            elif high >= stop:
                exit_reason = "손절"
                exit_price = stop
            
            if exit_reason:
                ret = (current_hedge['price'] - exit_price) / current_hedge['price'] * 100
                profit = current_hedge['invested'] * ret / 100
                hedge_trades.append({
                    'entry_date': current_hedge['entry_date'],
                    'exit_date': date,
                    'reason': exit_reason,
                    'invested': current_hedge['invested'],
                    'profit': profit,
                    'water': current_hedge['water']
                })
                current_hedge = None
        
        # 롱 청산
        if positions:
            total_qty = sum(CAPITAL / p['price'] for p in positions)
            avg_price = (len(positions) * CAPITAL) / total_qty
            ret = (price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = None
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
                
                water_counts.append(len(positions))
                max_water = max(max_water, len(positions))
                
                long_trades.append({
                    'entry_date': positions[0]['date'],
                    'exit_date': date,
                    'reason': exit_reason,
                    'water': len(positions),
                    'invested': invested,
                    'avg_price': avg_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'return': final_ret
                })
                
                # 숏도 청산
                if current_hedge:
                    s_ret = (current_hedge['price'] - price) / current_hedge['price'] * 100
                    s_profit = current_hedge['invested'] * s_ret / 100
                    hedge_trades.append({
                        'entry_date': current_hedge['entry_date'],
                        'exit_date': date,
                        'reason': '롱청산시',
                        'invested': current_hedge['invested'],
                        'profit': s_profit,
                        'water': current_hedge['water']
                    })
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
                    hedge_trades.append({
                        'entry_date': current_hedge['entry_date'],
                        'exit_date': date,
                        'reason': '업그레이드',
                        'invested': current_hedge['invested'],
                        'profit': s_profit,
                        'water': current_hedge['water']
                    })
                
                current_hedge = {
                    'entry_date': date,
                    'price': price,
                    'invested': num_buys * CAPITAL * HEDGE_RATIO,
                    'water': num_buys
                }
    
    # 현재 포지션
    current_long = None
    if positions:
        total_qty = sum(CAPITAL / p['price'] for p in positions)
        avg_price = (len(positions) * CAPITAL) / total_qty
        current_price = df_period['Close'].iloc[-1]
        unrealized = (current_price / avg_price - 1) * 100
        current_long = {
            'water': len(positions),
            'invested': len(positions) * CAPITAL,
            'avg_price': avg_price,
            'current_price': current_price,
            'unrealized': unrealized
        }
        max_water = max(max_water, len(positions))
    
    current_short = None
    if current_hedge:
        current_price = df_period['Close'].iloc[-1]
        unrealized = (current_hedge['price'] - current_price) / current_hedge['price'] * 100
        current_short = {
            'water': current_hedge['water'],
            'invested': current_hedge['invested'],
            'entry_price': current_hedge['price'],
            'current_price': current_price,
            'unrealized': unrealized
        }
    
    return long_trades, hedge_trades, max_water, water_counts, current_long, current_short

def analyze_period(df_period, period_name):
    buy_sigs, sell_sigs = find_signals(df_period)
    long_trades, hedge_trades, max_water, water_counts, current_long, current_short = simulate_detailed(
        df_period, buy_sigs, sell_sigs
    )
    
    # 롱 통계
    long_invested = sum(t['invested'] for t in long_trades)
    long_profit = sum(t['profit'] for t in long_trades)
    long_wins = sum(1 for t in long_trades if t['profit'] > 0)
    long_stoploss = sum(1 for t in long_trades if t['reason'] == '손절')
    avg_water = np.mean(water_counts) if water_counts else 0
    
    # 숏 통계
    short_invested = sum(t['invested'] for t in hedge_trades)
    short_profit = sum(t['profit'] for t in hedge_trades)
    short_wins = sum(1 for t in hedge_trades if t['profit'] > 0)
    short_stoploss = sum(1 for t in hedge_trades if t['reason'] == '손절')
    max_short_water = max([t['water'] for t in hedge_trades]) if hedge_trades else 0
    
    # ETH 변화
    eth_start = df_period['Close'].iloc[0]
    eth_end = df_period['Close'].iloc[-1]
    eth_change = (eth_end / eth_start - 1) * 100
    
    print(f'\n{"=" * 80}')
    print(f'📊 {period_name}')
    print(f'{"=" * 80}')
    print(f'기간: {df_period.index[0].strftime("%Y-%m-%d")} ~ {df_period.index[-1].strftime("%Y-%m-%d")}')
    print(f'ETH: ${eth_start:,.2f} → ${eth_end:,.2f} ({eth_change:+.0f}%)')
    
    print(f'\n🟢 롱 성과')
    print(f'   완료 거래: {len(long_trades)}회')
    print(f'   승률: {long_wins}/{len(long_trades)} ({long_wins/len(long_trades)*100:.0f}%)' if long_trades else '   승률: -')
    print(f'   손절: {long_stoploss}회')
    print(f'   총 투자금: ${long_invested:,}')
    print(f'   총 손익: ${long_profit:+,.0f}')
    print(f'   수익률: {long_profit/long_invested*100:+.2f}%' if long_invested > 0 else '   수익률: -')
    print(f'   최대 물타기: {max_water}회')
    print(f'   평균 물타기: {avg_water:.1f}회')
    
    print(f'\n🟣 숏 헷징 성과')
    print(f'   헷징 발동: {len(hedge_trades)}회')
    print(f'   승률: {short_wins}/{len(hedge_trades)} ({short_wins/len(hedge_trades)*100:.0f}%)' if hedge_trades else '   승률: -')
    print(f'   손절: {short_stoploss}회')
    print(f'   총 투자금: ${short_invested:,.0f}')
    print(f'   총 손익: ${short_profit:+,.0f}')
    print(f'   수익률: {short_profit/short_invested*100:+.2f}%' if short_invested > 0 else '   수익률: -')
    print(f'   최대 물타기시 헷징: {max_short_water}회')
    
    print(f'\n💰 총 성과')
    total_invested = long_invested + short_invested
    total_profit = long_profit + short_profit
    print(f'   총 투자금: ${total_invested:,.0f}')
    print(f'   총 손익: ${total_profit:+,.0f}')
    print(f'   금액 수익률: {total_profit/total_invested*100:+.2f}%' if total_invested > 0 else '   금액 수익률: -')
    
    if current_long:
        print(f'\n📍 현재 롱 포지션')
        print(f'   물타기: {current_long["water"]}회')
        print(f'   투자금: ${current_long["invested"]:,}')
        print(f'   평단가: ${current_long["avg_price"]:,.2f}')
        print(f'   현재가: ${current_long["current_price"]:,.2f}')
        print(f'   미실현: {current_long["unrealized"]:+.1f}%')
    
    if current_short:
        print(f'\n📍 현재 숏 포지션')
        print(f'   물타기시점: {current_short["water"]}회')
        print(f'   투자금: ${current_short["invested"]:,.0f}')
        print(f'   진입가: ${current_short["entry_price"]:,.2f}')
        print(f'   현재가: ${current_short["current_price"]:,.2f}')
        print(f'   미실현: {current_short["unrealized"]:+.1f}%')
    
    return {
        'long_trades': len(long_trades),
        'long_profit': long_profit,
        'short_trades': len(hedge_trades),
        'short_profit': short_profit,
        'total_profit': total_profit,
        'max_water': max_water
    }

# 헤더
print('=' * 80)
print('ETH 최종 전략 상세 성과 분석')
print('=' * 80)
print(f'\n📋 전략 파라미터:')
print(f'   RSI 매수: {RSI_BUY} 진입 → {RSI_BUY_EXIT} 탈출')
print(f'   RSI 매도: {RSI_SELL} 진입 → {RSI_SELL_EXIT} 탈출')
print(f'   손절: {STOP_LOSS}%')
print(f'   헷징: threshold={HEDGE_THRESHOLD}, upgrade={HEDGE_UPGRADE}, ratio={HEDGE_RATIO*100:.0f}%, profit={HEDGE_PROFIT}%, stop={HEDGE_STOP}%')

# 연도별 분석
years = [2020, 2021, 2022, 2023, 2024, 2025]
yearly_results = []

for year in years:
    df_year = df[(df.index >= f'{year}-01-01') & (df.index <= f'{year}-12-31')]
    if len(df_year) > 100:  # 충분한 데이터
        result = analyze_period(df_year, f'{year}년')
        result['year'] = year
        yearly_results.append(result)

# 전체 기간
print('\n' + '=' * 80)
analyze_period(df, '전체 기간 (5년)')

# 최근 2년
two_years_ago = df.index[-1] - timedelta(days=730)
df_2y = df[df.index >= two_years_ago]
analyze_period(df_2y, '최근 2년')

# 연도별 요약 테이블
print('\n' + '=' * 80)
print('📈 연도별 요약')
print('=' * 80)
print(f'{"연도":^6} | {"롱거래":^8} | {"롱손익":^12} | {"숏거래":^8} | {"숏손익":^12} | {"총손익":^12} | {"최대물타기":^10}')
print('-' * 80)

for r in yearly_results:
    print(f'{r["year"]:^6} | {r["long_trades"]:^8} | ${r["long_profit"]:+8,.0f} | '
          f'{r["short_trades"]:^8} | ${r["short_profit"]:+8,.0f} | ${r["total_profit"]:+8,.0f} | {r["max_water"]:^10}')

# 합계
print('-' * 80)
total_long = sum(r['long_profit'] for r in yearly_results)
total_short = sum(r['short_profit'] for r in yearly_results)
total = sum(r['total_profit'] for r in yearly_results)
max_w = max(r['max_water'] for r in yearly_results)
print(f'{"합계":^6} | {"":^8} | ${total_long:+8,.0f} | {"":^8} | ${total_short:+8,.0f} | ${total:+8,.0f} | {max_w:^10}')

