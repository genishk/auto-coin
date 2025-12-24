"""
데드크로스 숏 RSI 임계값 최적화 (일봉 5년)
"""

import pandas as pd
import sys
sys.path.insert(0, '.')
from src.data.cache import DataCache
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

cache = DataCache(cache_dir='data/cache', max_age_hours=24)
df = cache.get('BTC-USD_1d')
ti = TechnicalIndicators(load_config().get('indicators', {}))
df = ti.calculate_all(df)
df['MA40'] = df['Close'].rolling(window=40).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['dead_cross'] = df['MA40'] < df['MA200']

print('=' * 80)
print('🔍 데드크로스 숏 RSI 임계값 최적화 (일봉 5년)')
print('=' * 80)

# RSI 임계값 테스트
rsi_thresholds = [40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65]

results = []
for rsi_th in rsi_thresholds:
    signals = []
    for idx in range(200, len(df)):
        is_dead = df['dead_cross'].iloc[idx]
        rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        
        if pd.isna(rsi) or pd.isna(prev_rsi):
            continue
        
        if is_dead and prev_rsi > rsi_th and rsi <= rsi_th:
            signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx], 'idx': idx})
    
    # 시뮬레이션 (14일 보유, -15% 손절, profit_only)
    trades = []
    for s in signals:
        entry_idx = s['idx']
        entry_price = s['price']
        
        exited = False
        for hold in range(1, 30):  # 최대 30일까지 보유 가능
            if entry_idx + hold >= len(df):
                break
            
            exit_price = df['Close'].iloc[entry_idx + hold]
            ret = -((exit_price / entry_price - 1) * 100)  # 숏
            
            if ret <= -15:  # 손절
                trades.append({
                    'return': -15, 
                    'reason': 'stop', 
                    'year': df.index[entry_idx + hold].year
                })
                exited = True
                break
            elif hold >= 14 and ret > 0:  # 기간만료 + 수익
                trades.append({
                    'return': ret, 
                    'reason': 'expire', 
                    'year': df.index[entry_idx + hold].year
                })
                exited = True
                break
        
        # 30일 후에도 못 빠져나왔으면 강제 청산
        if not exited and entry_idx + 30 < len(df):
            exit_price = df['Close'].iloc[entry_idx + 30]
            ret = -((exit_price / entry_price - 1) * 100)
            trades.append({
                'return': ret, 
                'reason': 'force', 
                'year': df.index[entry_idx + 30].year
            })
    
    # 연도별 수익
    total = sum(t['return'] for t in trades)
    y2021 = sum(t['return'] for t in trades if t['year'] == 2021)
    y2022 = sum(t['return'] for t in trades if t['year'] == 2022)
    y2023 = sum(t['return'] for t in trades if t['year'] == 2023)
    y2024 = sum(t['return'] for t in trades if t['year'] == 2024)
    wins = len([t for t in trades if t['return'] > 0])
    win_rate = wins / len(trades) * 100 if trades else 0
    
    results.append({
        'rsi': rsi_th,
        'signals': len(signals),
        'trades': len(trades),
        'total': total,
        '2021': y2021,
        '2022': y2022,
        '2023': y2023,
        '2024': y2024,
        'win_rate': win_rate
    })

print(f"\n{'RSI':>5} | {'시그널':>6} | {'거래':>5} | {'전체':>10} | {'2021':>8} | {'2022':>8} | {'2023':>8} | {'2024':>8} | {'승률':>6}")
print('-' * 95)
for r in results:
    print(f"{r['rsi']:>5} | {r['signals']:>6} | {r['trades']:>5} | {r['total']:>+9.1f}% | {r['2021']:>+7.1f}% | {r['2022']:>+7.1f}% | {r['2023']:>+7.1f}% | {r['2024']:>+7.1f}% | {r['win_rate']:>5.1f}%")

# 최적값 찾기
print("\n" + "=" * 80)
print("🏆 최적값")
print("=" * 80)

best_total = max(results, key=lambda x: x['total'])
print(f"\n전체 기간 최적: RSI {best_total['rsi']}")
print(f"   총 수익: {best_total['total']:+.1f}%")
print(f"   승률: {best_total['win_rate']:.1f}%")

best_2022 = max(results, key=lambda x: x['2022'])
print(f"\n2022 하락장 최적: RSI {best_2022['rsi']}")
print(f"   2022년 수익: {best_2022['2022']:+.1f}%")

# 균형 점수 (전체 + 2022 - 2023 손실 최소화)
for r in results:
    r['balance'] = r['total'] + r['2022'] * 0.5 - abs(min(0, r['2023'])) * 0.3

best_balance = max(results, key=lambda x: x['balance'])
print(f"\n균형 최적 (하락장 중시): RSI {best_balance['rsi']}")
print(f"   전체: {best_balance['total']:+.1f}%, 2022: {best_balance['2022']:+.1f}%, 2023: {best_balance['2023']:+.1f}%")

