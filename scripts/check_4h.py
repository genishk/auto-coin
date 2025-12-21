"""
GitHub Actions용 4시간봉 시그널 체크 스크립트
- RSI 기반 매수/매도 시그널
- MA40/200 골든크로스 필터 (하락장 보호)
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators
from datetime import datetime
import os
import pandas as pd

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
    
    # 골든크로스용 이동평균선 추가
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    # 최신 데이터
    latest = df.iloc[-1]
    current_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    current_rsi = latest.get('rsi', 0)
    current_price = latest['Close']
    
    # 골든크로스 상태
    current_gc = latest['golden_cross'] if not pd.isna(latest['golden_cross']) else False
    ma40 = latest['MA40']
    ma200 = latest['MA200']
    
    # 가격 정보
    open_price = latest['Open']
    high_price = latest['High']
    low_price = latest['Low']
    close_price = latest['Close']
    
    # 시그널 체크
    buy_signal = False
    sell_signal = False
    
    # RSI 기준 (최적화된 값)
    rsi_oversold_threshold = 35
    rsi_buy_exit_threshold = 40
    
    rsi_overbought_threshold = 80
    rsi_sell_exit_threshold = 55
    
    # 최근 데이터에서 시그널 확인 (4시간봉 30개 = 5일)
    lookback = min(30, len(df))
    recent_df = df.iloc[-lookback:]
    
    # 매수 시그널 확인 (RSI 과매도 후 탈출 + 골든크로스)
    in_oversold = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        gc = recent_df['golden_cross'].iloc[i]
        
        if rsi < rsi_oversold_threshold:
            in_oversold = True
        elif in_oversold and rsi >= rsi_buy_exit_threshold:
            # 가장 최근 4시간봉이 탈출 시점인지 확인
            if i == len(recent_df) - 2:
                # 골든크로스 상태에서만 매수
                if gc:
                    buy_signal = True
            in_oversold = False
    
    # 현재 봉에서 탈출 확인
    if in_oversold and current_rsi >= rsi_buy_exit_threshold and current_gc:
        buy_signal = True
    
    # 매도 시그널 확인 (RSI 과매수 후 하락) - 골든크로스 무관
    in_overbought = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        if rsi > rsi_overbought_threshold:
            in_overbought = True
        elif in_overbought and rsi <= rsi_sell_exit_threshold:
            if i == len(recent_df) - 2:
                sell_signal = True
            in_overbought = False
    
    if in_overbought and current_rsi <= rsi_sell_exit_threshold:
        sell_signal = True
    
    # 결과 출력
    print('=' * 50)
    print('₿ Auto-Coin 4시간봉 리포트')
    print('=' * 50)
    print()
    print(f'📅 시간: {current_time} (UTC)')
    print()
    print('💰 가격 정보 (4시간봉)')
    print('-' * 40)
    print(f'시가: ${open_price:,.2f}')
    print(f'고가: ${high_price:,.2f}')
    print(f'저가: ${low_price:,.2f}')
    print(f'종가: ${close_price:,.2f}')
    print()
    print('📈 기술 지표')
    print('-' * 40)
    print(f'RSI: {current_rsi:.1f}')
    print(f'MA40: ${ma40:,.2f}' if not pd.isna(ma40) else 'MA40: N/A')
    print(f'MA200: ${ma200:,.2f}' if not pd.isna(ma200) else 'MA200: N/A')
    print(f'골든크로스: {"🟢 상승장" if current_gc else "🔴 하락장 (매수 차단)"}')
    print()
    print(f'매수 기준: RSI < {rsi_oversold_threshold} → RSI >= {rsi_buy_exit_threshold} (골든크로스 필수)')
    print(f'매도 기준: RSI > {rsi_overbought_threshold} → RSI <= {rsi_sell_exit_threshold}')
    print()
    print('🚨 시그널')
    print('-' * 40)
    
    if buy_signal:
        print(f'🟢 매수 시그널 발생!')
        print(f'   RSI 탈출 + 골든크로스 확인')
        print(f'   현재 가격: ${current_price:,.2f}')
    elif sell_signal:
        print(f'🔴 매도 시그널 발생!')
        print(f'   RSI가 {rsi_overbought_threshold} 이상에서 {rsi_sell_exit_threshold} 이하로 하락')
        print(f'   현재 가격: ${current_price:,.2f}')
    else:
        if not current_gc:
            print('📭 현재 시그널 없음 (하락장 - 매수 대기)')
        else:
            print('📭 현재 시그널 없음')
    
    print()
    print('=' * 50)
    
    # GitHub Actions 환경 변수 설정
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            if buy_signal:
                f.write(f'signal_type=buy\n')
                f.write(f'signal_price={current_price:,.2f}\n')
            elif sell_signal:
                f.write(f'signal_type=sell\n')
                f.write(f'signal_price={current_price:,.2f}\n')
            else:
                f.write('signal_type=none\n')
            f.write(f'current_time={current_time}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'open_price={open_price:,.2f}\n')
            f.write(f'high_price={high_price:,.2f}\n')
            f.write(f'low_price={low_price:,.2f}\n')
            f.write(f'close_price={close_price:,.2f}\n')
            f.write(f'golden_cross={"yes" if current_gc else "no"}\n')
            f.write(f'rsi_buy_threshold={rsi_oversold_threshold}\n')
            f.write(f'rsi_buy_exit={rsi_buy_exit_threshold}\n')
            f.write(f'rsi_sell_threshold={rsi_overbought_threshold}\n')
            f.write(f'rsi_sell_exit={rsi_sell_exit_threshold}\n')

if __name__ == '__main__':
    main()
