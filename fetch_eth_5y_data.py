"""
ETH 5년치 4시간봉 데이터 수집 (OKX)
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time

print('=' * 60)
print('ETH 5년치 4시간봉 데이터 수집 (OKX)')
print('=' * 60)

exchange = ccxt.okx()
symbol = 'ETH/USDT'
timeframe = '4h'

# 2020년 1월 1일부터 (5년)
start_date = datetime(2020, 1, 1)
end_date = datetime.now()

all_data = []
current = start_date

print(f'수집 기간: {start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}')
print()

while current < end_date:
    since = int(current.timestamp() * 1000)
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
        
        if not ohlcv:
            break
            
        all_data.extend(ohlcv)
        
        # 마지막 캔들 시간으로 업데이트
        last_ts = ohlcv[-1][0]
        current = datetime.fromtimestamp(last_ts / 1000) + timedelta(hours=4)
        
        first_dt = datetime.fromtimestamp(ohlcv[0][0]/1000).strftime("%Y-%m-%d")
        last_dt = datetime.fromtimestamp(last_ts/1000).strftime("%Y-%m-%d")
        print(f'  수집 중: {first_dt} ~ {last_dt} ({len(all_data)}개)')
        
        time.sleep(0.2)  # Rate limit
        
    except Exception as e:
        print(f'에러: {e}')
        break

# DataFrame 생성
df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('Date', inplace=True)
df.drop('timestamp', axis=1, inplace=True)

# 중복 제거
df = df[~df.index.duplicated(keep='first')]
df = df.sort_index()

print()
print('=' * 60)
print(f'✅ 수집 완료!')
print(f'   기간: {df.index[0]} ~ {df.index[-1]}')
print(f'   개수: {len(df)}개')
days = (df.index[-1] - df.index[0]).days
print(f'   기간: {days / 365:.1f}년')
print('=' * 60)

# 저장
df.to_csv('data/eth_4h_5y.csv')
print(f'저장: data/eth_4h_5y.csv')

print()
print(df.tail(3))

