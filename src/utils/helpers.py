"""유틸리티 함수"""

import yaml
from pathlib import Path


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """설정 파일 로드"""
    path = Path(config_path)
    
    if not path.exists():
        print(f"⚠️ 설정 파일 없음: {config_path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

