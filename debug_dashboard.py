"""
대시보드 로직 그대로 복사해서 디버깅
"""

import pandas as pd
import yfinance as yf
import sys
sys.path.insert(0, '.')
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 4시간봉 데이터 (대시보드와 동일)
ticker = yf.Ticker("BTC-USD")
df = ticker.history(period="2y", interval="1h")
df = df.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

ti = TechnicalIndicators(load_config().get('indicators', {}))
df = ti.calculate_all(df)
df['MA40'] = df['Close'].rolling(window=40).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['golden_cross'] = df['MA40'] > df['MA200']
df['dead_cross'] = df['MA40'] < df['MA200']

print('=' * 80)
print('🔍 대시보드 로직 디버깅')
print('=' * 80)
print(f"기간: {df.index[0]} ~ {df.index[-1]}")
print(f"총 봉: {len(df)}")

# 골든/데드크로스 비율
gc_count = df['golden_cross'].sum()
dc_count = df['dead_cross'].sum()
print(f"\n골든크로스: {gc_count}봉 ({gc_count/len(df)*100:.1f}%)")
print(f"데드크로스: {dc_count}봉 ({dc_count/len(df)*100:.1f}%)")

# 파라미터
SHORT_RSI_PEAK = 78
SHORT_RSI_ENTRY = 65
SHORT_LOOKBACK = 24
DC_RSI_THRESHOLD = 65


# ===== 기존 숏 시그널 (GC/DC 구분 없음) =====
def find_short_signals_original(df):
    signals = []
    for idx in range(SHORT_LOOKBACK, len(df)):
        recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
        had_peak = any(recent_rsi > SHORT_RSI_PEAK)
        if not had_peak:
            continue
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            continue
        if prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
            signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
    return signals


# ===== 새로운 숏 시그널 (GC/DC 구분) =====
def find_short_signals_new(df):
    signals = []
    gc_signals = []
    dc_signals = []
    
    for idx in range(SHORT_LOOKBACK, len(df)):
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        
        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            continue
        
        is_golden = df['golden_cross'].iloc[idx]
        is_dead = df['dead_cross'].iloc[idx]
        
        # 골든크로스: RSI peak 전략
        if is_golden:
            recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
            had_peak = any(recent_rsi > SHORT_RSI_PEAK)
            if had_peak and prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
                signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx], 'type': 'GC'})
                gc_signals.append(df.index[idx])
        
        # 데드크로스: RSI 하향 전략
        elif is_dead:
            if prev_rsi > DC_RSI_THRESHOLD and curr_rsi <= DC_RSI_THRESHOLD:
                signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx], 'type': 'DC'})
                dc_signals.append(df.index[idx])
    
    return signals, gc_signals, dc_signals


# 시그널 비교
original_signals = find_short_signals_original(df)
new_signals, gc_sigs, dc_sigs = find_short_signals_new(df)

print("\n" + "=" * 80)
print("📊 숏 시그널 비교")
print("=" * 80)
print(f"\n기존 숏 시그널: {len(original_signals)}개")
print(f"새로운 숏 시그널: {len(new_signals)}개 (GC: {len(gc_sigs)}, DC: {len(dc_sigs)})")

# 새로운 시그널이 기존보다 적은지/많은지
if len(new_signals) > len(original_signals):
    print(f"\n⚠️ 새 시그널이 {len(new_signals) - len(original_signals)}개 더 많음!")
elif len(new_signals) < len(original_signals):
    print(f"\n⚠️ 새 시그널이 {len(original_signals) - len(new_signals)}개 더 적음!")

# 기존에 있던 시그널이 새로운 방식에서 사라졌는지 확인
original_dates = set(s['confirm_date'] for s in original_signals)
new_dates = set(s['confirm_date'] for s in new_signals)

missing = original_dates - new_dates
added = new_dates - original_dates

print(f"\n기존에 있었지만 새로 사라진 시그널: {len(missing)}개")
print(f"새로 추가된 시그널: {len(added)}개")

if missing:
    print("\n사라진 시그널 (최근 10개):")
    for d in sorted(missing, reverse=True)[:10]:
        gc = df['golden_cross'].loc[d]
        print(f"  {d.strftime('%Y-%m-%d %H:%M')} - {'GC' if gc else 'DC'}")

if added:
    print("\n추가된 시그널 (최근 10개):")
    for d in sorted(added, reverse=True)[:10]:
        gc = df['golden_cross'].loc[d]
        print(f"  {d.strftime('%Y-%m-%d %H:%M')} - {'GC' if gc else 'DC'}")


# ===== 핵심 문제 확인 =====
print("\n" + "=" * 80)
print("🔴 핵심 문제 분석")
print("=" * 80)

# 기존 시그널 중 데드크로스 상태에서 발생한 것
original_in_dc = []
for s in original_signals:
    d = s['confirm_date']
    if d in df.index:
        gc = df['golden_cross'].loc[d]
        if not gc:  # 데드크로스
            original_in_dc.append(d)

print(f"\n기존 시그널 중 데드크로스 상태에서 발생: {len(original_in_dc)}개")
print(f"  → 이 시그널들이 새 전략에서는 DC 조건으로 대체됨")

# DC 상태에서 RSI peak 조건을 만족하지 않으면 시그널 누락
print("\n문제:")
print("  - 기존: DC 상태에서도 RSI peak 조건으로 숏 진입")
print("  - 새로운: DC 상태에서는 RSI 65 하향만으로 숏 진입")
print("  - DC 상태에서 RSI peak 조건 충족하는 기존 시그널이 사라짐!")

