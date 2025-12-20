"""코인 데이터 가져오기 모듈"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class CoinFetcher:
    """yfinance를 사용한 코인 데이터 fetcher"""
    
    def __init__(self, tickers: list):
        """
        Args:
            tickers: 코인 티커 리스트 (예: ['BTC-USD', 'ETH-USD'])
        """
        self.tickers = tickers
    
    def fetch(self, period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        코인 데이터 가져오기
        
        Args:
            period: 데이터 기간 (1y, 2y, 5y, 10y, max)
            interval: 봉 간격 (1h, 4h, 1d 등)
        
        Returns:
            {ticker: DataFrame} 형태의 딕셔너리
        """
        result = {}
        
        for ticker in self.tickers:
            interval_name = "일봉" if interval == "1d" else f"{interval}봉"
            print(f"📥 {ticker} {interval_name} 데이터 다운로드 중...")
            
            try:
                # yfinance로 데이터 가져오기
                coin = yf.Ticker(ticker)
                df = coin.history(period=period, interval=interval)
                
                if df.empty:
                    print(f"⚠️ {ticker}: 데이터 없음")
                    continue
                
                # 컬럼명 정리
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # 인덱스를 datetime으로 확실히 변환
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)  # timezone 제거
                
                unit = "개" if interval != "1d" else "일"
                print(f"✅ {ticker}: {len(df)}{unit} 데이터 로드 완료 ({interval_name})")
                print(f"   📅 {df.index[0]} ~ {df.index[-1]}")
                
                result[ticker] = df
                
            except Exception as e:
                print(f"❌ {ticker} 다운로드 실패: {e}")
        
        return result
    
    def fetch_single(self, ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """단일 코인 데이터 가져오기"""
        result = self.fetch([ticker] if ticker not in self.tickers else [ticker])
        return result.get(ticker)


def validate_data(df: pd.DataFrame, ticker: str) -> tuple:
    """
    데이터 검증 및 정리
    
    Returns:
        (cleaned_df, issues_list)
    """
    issues = []
    
    # 1. 결측치 확인
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"결측치 발견: {missing[missing > 0].to_dict()}")
        df = df.dropna()
    
    # 2. 중복 인덱스 확인
    if df.index.duplicated().any():
        issues.append("중복 날짜 발견 - 제거함")
        df = df[~df.index.duplicated(keep='first')]
    
    # 3. 정렬
    df = df.sort_index()
    
    # 4. 가격 유효성 (0 이하 제거)
    invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)
    if invalid_prices.any():
        issues.append(f"유효하지 않은 가격 {invalid_prices.sum()}개 제거")
        df = df[~invalid_prices]
    
    if issues:
        print(f"⚠️ {ticker} 데이터 이슈:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"✅ {ticker} 데이터 검증 통과")
    
    return df, issues


if __name__ == "__main__":
    # 테스트
    fetcher = CoinFetcher(['BTC-USD'])
    data = fetcher.fetch('5y')
    
    if 'BTC-USD' in data:
        df = data['BTC-USD']
        df, issues = validate_data(df, 'BTC-USD')
        
        print("\n📊 데이터 요약:")
        print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"총 {len(df)}일")
        print(f"\n최근 5일:")
        print(df.tail())

