"""데이터 캐싱 모듈"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import json


class DataCache:
    """데이터 캐싱 클래스"""
    
    def __init__(self, cache_dir: str = "data/cache", max_age_hours: int = 1):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            max_age_hours: 캐시 유효 시간 (코인은 24시간 거래라 1시간으로 짧게)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.metadata_file = self.cache_dir / "metadata.json"
    
    def _get_cache_path(self, ticker: str) -> Path:
        """캐시 파일 경로 반환"""
        # BTC-USD -> BTC_USD (파일명 호환)
        safe_ticker = ticker.replace('-', '_')
        return self.cache_dir / f"{safe_ticker}.parquet"
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict) -> None:
        """메타데이터 저장"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def is_valid(self, ticker: str) -> bool:
        """캐시가 유효한지 확인"""
        cache_path = self._get_cache_path(ticker)
        
        if not cache_path.exists():
            return False
        
        metadata = self._load_metadata()
        if ticker not in metadata:
            return False
        
        cached_time = datetime.fromisoformat(metadata[ticker]["cached_at"])
        if datetime.now() - cached_time > self.max_age:
            return False
        
        return True
    
    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 가져오기"""
        if not self.is_valid(ticker):
            return None
        
        cache_path = self._get_cache_path(ticker)
        
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"⚠️ {ticker} 캐시 로드 실패: {e}")
            return None
    
    def set(self, ticker: str, df: pd.DataFrame) -> None:
        """데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(ticker)
        
        try:
            df.to_parquet(cache_path)
            
            metadata = self._load_metadata()
            metadata[ticker] = {
                "cached_at": datetime.now().isoformat(),
                "rows": len(df),
                "start_date": str(df.index[0].date()) if len(df) > 0 else None,
                "end_date": str(df.index[-1].date()) if len(df) > 0 else None
            }
            self._save_metadata(metadata)
            
        except Exception as e:
            print(f"⚠️ {ticker} 캐시 저장 실패: {e}")
    
    def clear(self, ticker: str = None) -> None:
        """캐시 삭제"""
        if ticker:
            cache_path = self._get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
            
            metadata = self._load_metadata()
            if ticker in metadata:
                del metadata[ticker]
                self._save_metadata(metadata)
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
    
    def info(self) -> Dict:
        """캐시 정보 반환"""
        return self._load_metadata()

