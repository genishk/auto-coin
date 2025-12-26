"""
ETH 대시보드 함수로 직접 검증
"""
import sys
sys.path.insert(0, '.')

from dashboard_eth_4h import load_data, find_buy_signals, find_sell_signals, simulate_trades

# 데이터 로드
df = load_data('ETH-USD')

print('=' * 60)
print('대시보드 함수로 직접 테스트')
print('=' * 60)
print(f'데이터: {df.index[0]} ~ {df.index[-1]}')
print(f'데이터 개수: {len(df)}')
print()

# ETH 파라미터
rsi_oversold = 35
rsi_buy_exit = 40
rsi_overbought = 85
rsi_sell_exit = 55
stop_loss = -25
use_golden_cross = False

# 헷징 파라미터
use_hedge = True
hedge_threshold = 2
hedge_upgrade_interval = 5
hedge_ratio = 0.5
hedge_profit = 5
hedge_stop = -10

# 시그널 찾기
buy_signals = find_buy_signals(df, rsi_oversold, rsi_buy_exit, use_golden_cross)
sell_signals = find_sell_signals(df, rsi_overbought, rsi_sell_exit)

print(f'매수 시그널: {len(buy_signals)}개')
print(f'매도 시그널: {len(sell_signals)}개')
print()

# 시뮬레이션
trades, current_positions, hedge_trades, current_hedge = simulate_trades(
    df, buy_signals, sell_signals, stop_loss,
    use_hedge=use_hedge, hedge_threshold=hedge_threshold,
    hedge_upgrade_interval=hedge_upgrade_interval, hedge_ratio=hedge_ratio,
    hedge_profit=hedge_profit, hedge_stop=hedge_stop
)

# 결과
long_profit = sum(t['profit'] for t in trades)
long_invested = sum(t['invested'] for t in trades)
long_wins = sum(1 for t in trades if t['profit'] > 0)

short_profit = sum(h['profit'] for h in hedge_trades)
short_invested = sum(h.get('invested', 0) for h in hedge_trades)
short_wins = sum(1 for h in hedge_trades if h['profit'] > 0)

print('=' * 60)
print('대시보드 함수 결과')
print('=' * 60)
print(f'롱: {len(trades)}회')
print(f'   승률: {long_wins}/{len(trades)}')
print(f'   투자금: ${long_invested:,}')
print(f'   손익: ${long_profit:+,.0f}')
print()
print(f'숏: {len(hedge_trades)}회')
print(f'   승률: {short_wins}/{len(hedge_trades)}')
print(f'   투자금: ${short_invested:,.0f}')
print(f'   손익: ${short_profit:+,.0f}')
print()
print(f'총 손익: ${long_profit + short_profit:+,.0f}')

print()
print('=' * 60)
print('대시보드 표시값과 비교')
print('=' * 60)
print('대시보드 표시:')
print('   롱: 19회, +$1,704')
print('   숏: 26회, -$378')
print('   총: +$1,326')
print()
print('이 스크립트 결과:')
print(f'   롱: {len(trades)}회, ${long_profit:+,.0f}')
print(f'   숏: {len(hedge_trades)}회, ${short_profit:+,.0f}')
print(f'   총: ${long_profit + short_profit:+,.0f}')

