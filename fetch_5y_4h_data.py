"""
OKX에서 5년치 BTC 4시간봉 데이터 수집 및 전략 테스트
대시보드 함수 직접 사용
"""
import sys
sys.path.insert(0, '.')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# 대시보드 함수 import
from dashboard_4h_dual import (
    find_long_signals,
    find_long_exit_signals,
    find_short_signals,
    find_short_exit_signals,
    simulate_dual_trades
)

print("=" * 100)
print("📊 5년치 BTC 4시간봉 데이터 수집 및 전략 테스트")
print("=" * 100)


def fetch_all_4h_data(start_year=2020):
    """OKX에서 4시간봉 데이터 전체 수집"""
    exchange = ccxt.okx()
    exchange.load_markets()
    
    symbol = 'BTC/USDT'
    timeframe = '4h'
    
    # 시작일 설정
    start_date = f'{start_year}-01-01T00:00:00Z'
    since = exchange.parse8601(start_date)
    
    print(f"\n📅 수집 시작: {start_date}")
    print(f"🔄 {symbol} {timeframe} 데이터 수집 중...")
    
    all_candles = []
    request_count = 0
    
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            request_count += 1
            
            # 진행 상황 출력
            last_date = datetime.fromtimestamp(candles[-1][0] / 1000)
            print(f"   요청 {request_count}: {len(candles)}개 캔들 (~ {last_date.strftime('%Y-%m-%d')})")
            
            # 마지막 캔들이 현재와 가까우면 종료
            if candles[-1][0] >= exchange.milliseconds() - 4 * 60 * 60 * 1000:
                break
            
            # 다음 요청을 위한 시작점 업데이트
            since = candles[-1][0] + 1
            
            # Rate limit 방지
            time.sleep(0.3)
            
        except Exception as e:
            print(f"   ⚠️ 오류 발생: {e}")
            time.sleep(1)
            continue
    
    print(f"\n✅ 총 {len(all_candles)}개 캔들 수집 완료!")
    
    # DataFrame 변환
    df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    
    # 중복 제거
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    return df


def add_technical_indicators(df):
    """기술적 지표 추가 (대시보드와 동일)"""
    # RSI 계산
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # MA 계산 (대시보드와 동일: MA100, MA200)
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Golden Cross / Dead Cross
    df['golden_cross'] = df['MA100'] > df['MA200']
    df['dead_cross'] = df['MA100'] < df['MA200']
    
    return df


def run_strategy_test(df, name=""):
    """대시보드 함수로 전략 테스트"""
    # 파라미터 (대시보드 기본값)
    LONG_RSI_OVERSOLD = 35
    LONG_RSI_EXIT = 40
    LONG_RSI_OVERBOUGHT = 80
    LONG_RSI_SELL = 55
    LONG_STOP_LOSS = -25
    
    SHORT_RSI_PEAK = 78
    SHORT_RSI_ENTRY = 65
    SHORT_RSI_EXIT = 45
    SHORT_STOP_LOSS = -15
    SHORT_MAX_HOLD = 42
    SHORT_LOOKBACK = 24
    DC_RSI_THRESHOLD = 55
    SHORT_MAX_ENTRIES = 4
    
    # 시그널 계산
    long_signals = find_long_signals(df, LONG_RSI_OVERSOLD, LONG_RSI_EXIT, True)
    long_exit_signals = find_long_exit_signals(df, LONG_RSI_OVERBOUGHT, LONG_RSI_SELL)
    short_signals = find_short_signals(df, SHORT_RSI_PEAK, SHORT_RSI_ENTRY, SHORT_LOOKBACK, DC_RSI_THRESHOLD)
    short_exit_signals = find_short_exit_signals(df, LONG_RSI_OVERSOLD, SHORT_RSI_EXIT)
    
    # 시뮬레이션
    trades, _ = simulate_dual_trades(
        df, long_signals, long_exit_signals,
        short_signals, short_exit_signals,
        LONG_STOP_LOSS, SHORT_STOP_LOSS, SHORT_MAX_HOLD, SHORT_MAX_ENTRIES
    )
    
    # 결과 집계
    long_trades = [t for t in trades if t['type'] == 'long']
    short_trades = [t for t in trades if t['type'] == 'short']
    
    total_return = sum(t['return'] for t in trades)
    long_return = sum(t['return'] for t in long_trades)
    short_return = sum(t['return'] for t in short_trades)
    
    long_wins = len([t for t in long_trades if t['return'] > 0])
    short_wins = len([t for t in short_trades if t['return'] > 0])
    
    return {
        'name': name,
        'total_return': total_return,
        'long_return': long_return,
        'short_return': short_return,
        'total_trades': len(trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': long_wins / len(long_trades) * 100 if long_trades else 0,
        'short_win_rate': short_wins / len(short_trades) * 100 if short_trades else 0,
        'trades': trades
    }


def analyze_by_year(trades, df):
    """연도별 성과 분석"""
    results = {}
    
    for trade in trades:
        year = trade['exit_date'].year
        if year not in results:
            results[year] = {'long': 0, 'short': 0, 'long_count': 0, 'short_count': 0}
        
        if trade['type'] == 'long':
            results[year]['long'] += trade['return']
            results[year]['long_count'] += 1
        else:
            results[year]['short'] += trade['return']
            results[year]['short_count'] += 1
    
    return results


# ===== 메인 실행 =====
if __name__ == "__main__":
    # 캐시 파일 확인
    cache_file = "data/btc_4h_5y.csv"
    
    if os.path.exists(cache_file):
        print(f"\n📁 캐시 파일 발견: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"   로드된 데이터: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")
    else:
        # 데이터 수집
        df = fetch_all_4h_data(start_year=2020)
        
        # 기술적 지표 추가
        df = add_technical_indicators(df)
        
        # 캐시 저장
        os.makedirs("data", exist_ok=True)
        df.to_csv(cache_file)
        print(f"\n💾 캐시 저장: {cache_file}")
    
    # 데이터 요약
    print(f"\n📊 데이터 요약:")
    print(f"   기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"   총 봉 수: {len(df):,}개")
    print(f"   약 {len(df) * 4 / 24 / 365:.1f}년치 데이터")
    
    # NaN 제거 (MA 계산으로 인한)
    df_clean = df.dropna()
    print(f"   유효 봉 수: {len(df_clean):,}개 (MA 계산 후)")
    
    # ===== 전략 테스트 =====
    print("\n" + "=" * 100)
    print("📈 전략 테스트 (대시보드 함수 직접 사용)")
    print("=" * 100)
    
    result = run_strategy_test(df_clean, "5년치 4시간봉")
    
    print(f"\n📊 전체 성과:")
    print(f"   총 누적 수익률: {result['total_return']:+.1f}%")
    print(f"   롱 누적: {result['long_return']:+.1f}% ({result['long_trades']}회, 승률 {result['long_win_rate']:.1f}%)")
    print(f"   숏 누적: {result['short_return']:+.1f}% ({result['short_trades']}회, 승률 {result['short_win_rate']:.1f}%)")
    
    # 연도별 분석
    yearly = analyze_by_year(result['trades'], df_clean)
    
    print(f"\n📅 연도별 성과:")
    print(f"{'연도':>6} | {'롱':>10} | {'숏':>10} | {'합계':>10} | {'롱거래':>6} | {'숏거래':>6}")
    print("-" * 65)
    
    for year in sorted(yearly.keys()):
        y = yearly[year]
        total = y['long'] + y['short']
        print(f"{year:>6} | {y['long']:>+9.1f}% | {y['short']:>+9.1f}% | {total:>+9.1f}% | {y['long_count']:>6} | {y['short_count']:>6}")
    
    print("\n" + "=" * 100)
    print("✅ 테스트 완료!")
    print("=" * 100)

