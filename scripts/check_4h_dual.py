"""
GitHub Actions용 4시간봉 롱/숏 듀얼 시그널 체크 스크립트
- 롱: RSI 과매도 탈출 + 골든크로스 (MA100/200)
- 숏: GC에서 RSI peak 하향, DC에서 RSI 55 하향
- 하락장 방어 최적화 전략
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators
from datetime import datetime
import os
import pandas as pd

# 전략 파라미터
LONG_RSI_OVERSOLD = 35
LONG_RSI_EXIT = 40
LONG_RSI_OVERBOUGHT = 80
LONG_RSI_SELL = 55

SHORT_RSI_PEAK = 78
SHORT_RSI_ENTRY = 65
SHORT_LOOKBACK = 24
DC_RSI_THRESHOLD = 55

SHORT_RSI_EXIT = 45


def main():
    ticker = 'BTC-USD'
    
    # 데이터 로드 (4시간봉, 2년)
    cache = DataCache(cache_dir='data/cache_4h', max_age_hours=4)
    df = cache.get(f'{ticker}_4h')
    
    if df is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        df = data[ticker]
        df, _ = validate_data(df, ticker)
        cache.set(f'{ticker}_4h', df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators()
    df = ti.calculate_all(df)
    
    # MA100/200 (하락장 방어 최적화)
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA100'] > df['MA200']
    df['dead_cross'] = df['MA100'] < df['MA200']
    
    # 최신 데이터
    latest = df.iloc[-1]
    current_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    current_rsi = latest.get('rsi', 0)
    current_price = latest['Close']
    
    # 추세 상태
    current_gc = latest['golden_cross'] if not pd.isna(latest['golden_cross']) else False
    current_dc = latest['dead_cross'] if not pd.isna(latest['dead_cross']) else False
    ma100 = latest['MA100']
    ma200 = latest['MA200']
    
    # 가격 정보
    open_price = latest['Open']
    high_price = latest['High']
    low_price = latest['Low']
    close_price = latest['Close']
    
    # 시그널 체크
    long_entry_signal = False
    long_exit_signal = False
    short_entry_signal = False
    short_exit_signal = False
    
    # 최근 데이터
    lookback = min(SHORT_LOOKBACK + 5, len(df))
    recent_df = df.iloc[-lookback:]
    
    # ===== 롱 진입 시그널 (RSI 과매도 탈출 + 골든크로스) =====
    in_oversold = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        gc = recent_df['golden_cross'].iloc[i]
        
        if rsi < LONG_RSI_OVERSOLD:
            in_oversold = True
        elif in_oversold and rsi >= LONG_RSI_EXIT:
            if i == len(recent_df) - 2 and gc:
                long_entry_signal = True
            in_oversold = False
    
    if in_oversold and current_rsi >= LONG_RSI_EXIT and current_gc:
        long_entry_signal = True
    
    # ===== 롱 청산 시그널 (RSI 과매수 후 하락) =====
    in_overbought = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        if rsi > LONG_RSI_OVERBOUGHT:
            in_overbought = True
        elif in_overbought and rsi <= LONG_RSI_SELL:
            if i == len(recent_df) - 2:
                long_exit_signal = True
            in_overbought = False
    
    if in_overbought and current_rsi <= LONG_RSI_SELL:
        long_exit_signal = True
    
    # ===== 숏 진입 시그널 =====
    # GC: RSI peak 후 하향
    if current_gc:
        recent_rsi = df['rsi'].iloc[-SHORT_LOOKBACK-1:-1]
        had_peak = any(recent_rsi > SHORT_RSI_PEAK)
        prev_rsi = df['rsi'].iloc[-2]
        
        if had_peak and prev_rsi > SHORT_RSI_ENTRY and current_rsi <= SHORT_RSI_ENTRY:
            short_entry_signal = True
    
    # DC: RSI threshold 하향 (하락장 방어)
    elif current_dc:
        prev_rsi = df['rsi'].iloc[-2]
        if prev_rsi > DC_RSI_THRESHOLD and current_rsi <= DC_RSI_THRESHOLD:
            short_entry_signal = True
    
    # ===== 숏 청산 시그널 (RSI 과매도 후 탈출) =====
    in_oversold_short = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        if rsi < LONG_RSI_OVERSOLD:
            in_oversold_short = True
        elif in_oversold_short and rsi >= SHORT_RSI_EXIT:
            if i == len(recent_df) - 2:
                short_exit_signal = True
            in_oversold_short = False
    
    if in_oversold_short and current_rsi >= SHORT_RSI_EXIT:
        short_exit_signal = True
    
    # 추세 상태 문자열
    trend_status = "🟢 상승장 (GC)" if current_gc else "🔴 하락장 (DC)"
    
    # 결과 출력
    print('=' * 55)
    print('🔄 Auto-Coin 롱/숏 듀얼 4시간봉 리포트')
    print('=' * 55)
    print()
    print(f'📅 시간: {current_time} (UTC)')
    print()
    print('💰 가격 정보 (4시간봉)')
    print('-' * 45)
    print(f'시가: ${open_price:,.2f}')
    print(f'고가: ${high_price:,.2f}')
    print(f'저가: ${low_price:,.2f}')
    print(f'종가: ${close_price:,.2f}')
    print()
    print('📈 기술 지표')
    print('-' * 45)
    print(f'RSI: {current_rsi:.1f}')
    print(f'MA100: ${ma100:,.2f}' if not pd.isna(ma100) else 'MA100: N/A')
    print(f'MA200: ${ma200:,.2f}' if not pd.isna(ma200) else 'MA200: N/A')
    print(f'추세: {trend_status}')
    print()
    print('📋 전략 기준')
    print('-' * 45)
    print(f'🟢 롱 진입: RSI < {LONG_RSI_OVERSOLD} → RSI >= {LONG_RSI_EXIT} + GC')
    print(f'🟢 롱 청산: RSI > {LONG_RSI_OVERBOUGHT} → RSI <= {LONG_RSI_SELL}')
    if current_gc:
        print(f'🔴 숏 진입: RSI > {SHORT_RSI_PEAK} (최근 {SHORT_LOOKBACK}봉) → RSI <= {SHORT_RSI_ENTRY}')
    else:
        print(f'🔴 숏 진입: RSI > {DC_RSI_THRESHOLD} → RSI <= {DC_RSI_THRESHOLD} (DC)')
    print(f'🔴 숏 청산: RSI < {LONG_RSI_OVERSOLD} → RSI >= {SHORT_RSI_EXIT}')
    print()
    print('🚨 시그널')
    print('-' * 45)
    
    signal_type = 'none'
    signal_detail = ''
    
    if long_entry_signal:
        signal_type = 'long_entry'
        signal_detail = f'🟢 롱 진입 시그널!\n   RSI 탈출 + 골든크로스\n   가격: ${current_price:,.2f}'
        print(signal_detail)
    elif long_exit_signal:
        signal_type = 'long_exit'
        signal_detail = f'🟡 롱 청산 시그널!\n   RSI {LONG_RSI_OVERBOUGHT}+ 후 {LONG_RSI_SELL} 하향\n   가격: ${current_price:,.2f}'
        print(signal_detail)
    elif short_entry_signal:
        signal_type = 'short_entry'
        if current_gc:
            signal_detail = f'🔴 숏 진입 시그널! (GC)\n   RSI peak 후 하향\n   가격: ${current_price:,.2f}'
        else:
            signal_detail = f'🔴 숏 진입 시그널! (DC)\n   RSI {DC_RSI_THRESHOLD} 하향 돌파\n   가격: ${current_price:,.2f}'
        print(signal_detail)
    elif short_exit_signal:
        signal_type = 'short_exit'
        signal_detail = f'🟡 숏 청산 시그널!\n   RSI 과매도 탈출\n   가격: ${current_price:,.2f}'
        print(signal_detail)
    else:
        print('📭 현재 시그널 없음')
    
    print()
    print('=' * 55)
    
    # GitHub Actions 환경 변수 설정
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f'signal_type={signal_type}\n')
            f.write(f'signal_price={current_price:,.2f}\n')
            f.write(f'current_time={current_time}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'open_price={open_price:,.2f}\n')
            f.write(f'high_price={high_price:,.2f}\n')
            f.write(f'low_price={low_price:,.2f}\n')
            f.write(f'close_price={close_price:,.2f}\n')
            f.write(f'trend={"GC" if current_gc else "DC"}\n')
            f.write(f'ma100={ma100:,.2f}\n' if not pd.isna(ma100) else 'ma100=N/A\n')
            f.write(f'ma200={ma200:,.2f}\n' if not pd.isna(ma200) else 'ma200=N/A\n')


if __name__ == '__main__':
    main()

