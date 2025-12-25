"""
숏 헷징 손절 라인 최적화
5년 데이터 + 구간별 분석
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, '.')

from dashboard_4h import find_buy_signals, find_sell_signals, simulate_trades

def add_indicators(df):
    """지표 추가"""
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
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def test_strategy(df, hedge_stop):
    """전략 테스트"""
    # 기본 설정
    RSI_OVERSOLD, RSI_BUY_EXIT = 35, 40
    RSI_OVERBOUGHT, RSI_SELL_EXIT = 80, 55
    STOP_LOSS = -25
    USE_GOLDEN_CROSS = False
    
    HEDGE_THRESHOLD = 2
    HEDGE_UPGRADE_INTERVAL = 3
    HEDGE_RATIO = 1.0
    HEDGE_PROFIT = 8
    CAPITAL = 1000
    
    buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
    sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
    
    # hedge_stop이 None이면 아주 큰 값으로 (실질적으로 손절 없음)
    actual_stop = hedge_stop if hedge_stop is not None else -999
    
    trades, _, hedge_trades, _ = simulate_trades(
        df, buy_signals, sell_signals, STOP_LOSS,
        use_hedge=True, hedge_threshold=HEDGE_THRESHOLD,
        hedge_upgrade_interval=HEDGE_UPGRADE_INTERVAL, hedge_ratio=HEDGE_RATIO,
        hedge_profit=HEDGE_PROFIT, hedge_stop=actual_stop
    )
    
    # 롱 성과
    long_invested = sum(t['num_buys'] * CAPITAL for t in trades)
    long_profit = sum(t['num_buys'] * CAPITAL * t['return'] / 100 for t in trades)
    
    # 숏 성과
    if hedge_trades:
        short_invested = sum(h.get('invested', h['long_num_buys'] * CAPITAL) for h in hedge_trades)
        short_profit = sum(h.get('invested', h['long_num_buys'] * CAPITAL) * h['return'] / 100 for h in hedge_trades)
        short_wins = len([h for h in hedge_trades if h['return'] > 0])
        short_count = len(hedge_trades)
    else:
        short_invested, short_profit, short_wins, short_count = 0, 0, 0, 0
    
    total_profit = long_profit + short_profit
    
    return {
        'long_trades': len(trades),
        'long_invested': long_invested,
        'long_profit': long_profit,
        'long_return': long_profit / long_invested * 100 if long_invested > 0 else 0,
        'short_count': short_count,
        'short_invested': short_invested,
        'short_profit': short_profit,
        'short_return': short_profit / short_invested * 100 if short_invested > 0 else 0,
        'short_wins': short_wins,
        'short_win_rate': short_wins / short_count * 100 if short_count > 0 else 0,
        'total_profit': total_profit,
        'total_return': total_profit / long_invested * 100 if long_invested > 0 else 0
    }

# 데이터 로드
print("데이터 로딩 중...")
df_full = pd.read_csv('data/btc_4h_5y.csv', index_col=0, parse_dates=True)
df_full = add_indicators(df_full)
print(f"전체 데이터: {df_full.index[0]} ~ {df_full.index[-1]}")

# 구간 정의
periods = {
    '전체 5년': (df_full.index[0], df_full.index[-1]),
    '2020-2021 (상승장)': ('2020-01-01', '2021-12-31'),
    '2022 (하락장)': ('2022-01-01', '2022-12-31'),
    '2023 (횡보/회복)': ('2023-01-01', '2023-12-31'),
    '2024-현재 (상승장)': ('2024-01-01', df_full.index[-1])
}

# 손절 라인 옵션
stop_options = [
    None,  # 손절 없음
    -5, -10, -15, -20, -25, -30, -40, -50
]

print("\n" + "="*100)
print("📊 숏 헷징 손절 라인 최적화 (5년 데이터)")
print("="*100)
print(f"헷징 설정: 2회 시작, 3회마다 업그레이드, 100% 비율, 8% 익절")
print("="*100)

for period_name, (start, end) in periods.items():
    print(f"\n{'='*100}")
    print(f"📅 {period_name}")
    print(f"{'='*100}")
    
    # 구간 데이터 추출
    df_period = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()
    
    if len(df_period) < 200:
        print("   데이터 부족 (200캔들 미만)")
        continue
    
    btc_start = df_period['Close'].iloc[0]
    btc_end = df_period['Close'].iloc[-1]
    btc_change = (btc_end / btc_start - 1) * 100
    print(f"BTC 변동: ${btc_start:,.0f} → ${btc_end:,.0f} ({btc_change:+.1f}%)")
    print()
    
    results = []
    
    for stop in stop_options:
        result = test_strategy(df_period, stop)
        result['stop'] = stop
        results.append(result)
    
    # 헤더
    print(f"{'손절라인':>10} | {'숏거래':>6} | {'숏승률':>6} | {'숏손익':>12} | {'숏수익률':>8} | {'총손익':>12} | {'총수익률':>8}")
    print("-" * 90)
    
    # 결과 출력
    for r in results:
        stop_str = "없음" if r['stop'] is None else f"{r['stop']}%"
        print(f"{stop_str:>10} | {r['short_count']:>6} | {r['short_win_rate']:>5.0f}% | ${r['short_profit']:>+10,.0f} | {r['short_return']:>+7.1f}% | ${r['total_profit']:>+10,.0f} | {r['total_return']:>+7.1f}%")
    
    # 최적 찾기
    best = max(results, key=lambda x: x['total_profit'])
    print()
    print(f"🏆 최적 손절라인: {'없음' if best['stop'] is None else f'{best["stop"]}%'} → 총 수익 ${best['total_profit']:+,.0f} ({best['total_return']:+.1f}%)")

# 전체 기간 상세 분석
print("\n" + "="*100)
print("📊 전체 5년 상세 분석")
print("="*100)

for stop in stop_options:
    result = test_strategy(df_full, stop)
    stop_str = "없음" if stop is None else f"{stop}%"
    
    print(f"\n{'─'*50}")
    print(f"손절라인: {stop_str}")
    print(f"{'─'*50}")
    print(f"  🟢 롱: {result['long_trades']}회, 투자금 ${result['long_invested']:,.0f}, 손익 ${result['long_profit']:+,.0f} ({result['long_return']:+.1f}%)")
    print(f"  🟣 숏: {result['short_count']}회, 승률 {result['short_win_rate']:.0f}%, 손익 ${result['short_profit']:+,.0f} ({result['short_return']:+.1f}%)")
    print(f"  💰 총: ${result['total_profit']:+,.0f} ({result['total_return']:+.1f}%)")

