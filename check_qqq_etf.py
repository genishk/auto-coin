"""
QQQ 관련 ETF 확인
"""
import yfinance as yf

tickers = ['QQQ', 'TQQQ', 'QLD', 'SQQQ', 'QID', 'PSQ', 'QQQM', 'QQQE']

print("=" * 70)
print("QQQ / Nasdaq-100 ETF List")
print("=" * 70)

for t in tickers:
    try:
        ticker = yf.Ticker(t)
        info = ticker.info
        name = info.get('longName', info.get('shortName', 'N/A'))
        price = info.get('regularMarketPrice', 'N/A')
        print(f"\n{t}:")
        print(f"  Name: {name}")
        print(f"  Price: ${price}")
    except Exception as e:
        print(f"\n{t}: Error - {e}")

