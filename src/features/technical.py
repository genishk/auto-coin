"""기술적 지표 계산 모듈"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 지표 설정 (settings.yaml의 indicators 섹션)
        """
        self.config = config or {}
        
        # 기본값 설정
        self.rsi_period = self.config.get('rsi', {}).get('period', 14)
        self.macd_fast = self.config.get('macd', {}).get('fast', 12)
        self.macd_slow = self.config.get('macd', {}).get('slow', 26)
        self.macd_signal = self.config.get('macd', {}).get('signal', 9)
        self.bb_period = self.config.get('bollinger', {}).get('period', 20)
        self.bb_std = self.config.get('bollinger', {}).get('std', 2)
        self.ma_short = self.config.get('moving_averages', {}).get('short', 20)
        self.ma_medium = self.config.get('moving_averages', {}).get('medium', 50)
        self.ma_long = self.config.get('moving_averages', {}).get('long', 200)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 계산"""
        df = df.copy()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(
            df['Close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger(
            df['Close'], self.bb_period, self.bb_std
        )
        
        # Moving Averages
        df['ma_short'] = df['Close'].rolling(window=self.ma_short).mean()
        df['ma_medium'] = df['Close'].rolling(window=self.ma_medium).mean()
        df['ma_long'] = df['Close'].rolling(window=self.ma_long).mean()
        
        # 추가 지표
        df['momentum'] = df['Close'].pct_change(periods=10) * 100
        df['volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean() * 100
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int):
        """MACD 계산"""
        exp_fast = prices.ewm(span=fast, adjust=False).mean()
        exp_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = exp_fast - exp_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger(self, prices: pd.Series, period: int, std_dev: float):
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower


if __name__ == "__main__":
    # 테스트
    from src.data.fetcher import CoinFetcher, validate_data
    
    fetcher = CoinFetcher(['BTC-USD'])
    data = fetcher.fetch('5y')
    df = data['BTC-USD']
    df, _ = validate_data(df, 'BTC-USD')
    
    ti = TechnicalIndicators()
    df = ti.calculate_all(df)
    
    print("\n📊 기술 지표 계산 완료:")
    print(df[['Close', 'rsi', 'macd', 'ma_short', 'momentum']].tail(10))

