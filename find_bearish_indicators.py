"""
물타기 많은 시기를 감지할 수 있는 다양한 하락장 지표 테스트
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from dashboard_4h import (
    find_buy_signals,
    find_sell_signals,
    simulate_trades
)

print("=" * 100)
print("📊 하락장 감지 지표 종합 테스트")
print("=" * 100)

# 파라미터 (GC OFF)
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25
USE_GOLDEN_CROSS = False

# 데이터 로드
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = df.dropna()
print(f"데이터: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)\n")

# ===== 다양한 지표 계산 =====
print("📈 지표 계산 중...")

# 기본 MA
for period in [20, 50, 100, 200]:
    df[f'MA{period}'] = df['Close'].rolling(period).mean()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp12 = df['Close'].ewm(span=12).mean()
exp26 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp12 - exp26
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# 볼린저 밴드
df['BB_mid'] = df['Close'].rolling(20).mean()
df['BB_std'] = df['Close'].rolling(20).std()
df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# ATR (변동성)
high_low = df['High'] - df['Low']
high_close = abs(df['High'] - df['Close'].shift())
low_close = abs(df['Low'] - df['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = tr.rolling(14).mean()
df['ATR_pct'] = df['ATR'] / df['Close'] * 100

# 최근 고점 대비 하락률
for lookback in [30, 60, 90, 120]:
    df[f'high_{lookback}'] = df['High'].rolling(lookback).max()
    df[f'drawdown_{lookback}'] = (df['Close'] - df[f'high_{lookback}']) / df[f'high_{lookback}'] * 100

# N일 연속 하락
df['daily_return'] = df['Close'].pct_change()
df['consecutive_down'] = 0
consecutive = 0
for i in range(len(df)):
    if df['daily_return'].iloc[i] < 0:
        consecutive += 1
    else:
        consecutive = 0
    df.iloc[i, df.columns.get_loc('consecutive_down')] = consecutive

# 가격 vs MA 위치
df['below_MA50'] = df['Close'] < df['MA50']
df['below_MA100'] = df['Close'] < df['MA100']
df['below_MA200'] = df['Close'] < df['MA200']

# 데드크로스
df['DC_50_200'] = df['MA50'] < df['MA200']
df['DC_100_200'] = df['MA100'] < df['MA200']

# RSI 추세
df['RSI_MA'] = df['RSI'].rolling(14).mean()
df['RSI_falling'] = df['RSI'] < df['RSI_MA']

# MACD 음전환
df['MACD_negative'] = df['MACD'] < 0
df['MACD_hist_negative'] = df['MACD_hist'] < 0

# 시그널 및 시뮬레이션
buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, USE_GOLDEN_CROSS)
sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
trades, _ = simulate_trades(df, buy_signals, sell_signals, STOP_LOSS)

# 물타기 5회 이상 거래
heavy_trades = [t for t in trades if t['num_buys'] >= 5]
print(f"총 거래: {len(trades)}회, 물타기 5회+: {len(heavy_trades)}회\n")

# ===== 각 지표가 물타기 시기를 얼마나 잘 감지하는지 =====
print("=" * 100)
print("📊 지표별 감지율 분석")
print("=" * 100)

# 각 거래의 시작 시점에서 지표 상태 확인
def analyze_indicator_at_trade_start(trades, df, indicator_name, condition_func):
    """거래 시작 시점에서 지표 조건이 충족되었는지 확인"""
    detected = 0
    for trade in trades:
        entry_date = trade['entry_dates'][0]
        # 진입 시점 또는 그 직전 데이터
        idx = df.index.get_indexer([entry_date], method='ffill')[0]
        if idx >= 0 and idx < len(df):
            if condition_func(df, idx):
                detected += 1
    return detected, len(trades), detected / len(trades) * 100 if trades else 0

# 다양한 지표 조건 정의
indicators = {
    # 데드크로스 기반
    "데드크로스 MA50/200": lambda df, i: df['DC_50_200'].iloc[i] if i < len(df) else False,
    "데드크로스 MA100/200": lambda df, i: df['DC_100_200'].iloc[i] if i < len(df) else False,
    
    # 가격 vs MA
    "가격 < MA50": lambda df, i: df['below_MA50'].iloc[i] if i < len(df) else False,
    "가격 < MA100": lambda df, i: df['below_MA100'].iloc[i] if i < len(df) else False,
    "가격 < MA200": lambda df, i: df['below_MA200'].iloc[i] if i < len(df) else False,
    
    # 고점 대비 하락률
    "고점대비 -10% 이상": lambda df, i: df['drawdown_60'].iloc[i] <= -10 if i < len(df) else False,
    "고점대비 -15% 이상": lambda df, i: df['drawdown_60'].iloc[i] <= -15 if i < len(df) else False,
    "고점대비 -20% 이상": lambda df, i: df['drawdown_60'].iloc[i] <= -20 if i < len(df) else False,
    "고점대비 -25% 이상": lambda df, i: df['drawdown_60'].iloc[i] <= -25 if i < len(df) else False,
    
    # RSI 기반
    "RSI < 40": lambda df, i: df['RSI'].iloc[i] < 40 if i < len(df) else False,
    "RSI < 50": lambda df, i: df['RSI'].iloc[i] < 50 if i < len(df) else False,
    "RSI 하락추세": lambda df, i: df['RSI_falling'].iloc[i] if i < len(df) else False,
    
    # MACD 기반
    "MACD < 0": lambda df, i: df['MACD_negative'].iloc[i] if i < len(df) else False,
    "MACD 히스토그램 < 0": lambda df, i: df['MACD_hist_negative'].iloc[i] if i < len(df) else False,
    
    # 볼린저밴드
    "BB 하단 근처 (<20%)": lambda df, i: df['BB_pct'].iloc[i] < 0.2 if i < len(df) else False,
    "BB 하단 돌파 (<0%)": lambda df, i: df['BB_pct'].iloc[i] < 0 if i < len(df) else False,
    
    # 변동성
    "ATR > 3%": lambda df, i: df['ATR_pct'].iloc[i] > 3 if i < len(df) else False,
    "ATR > 4%": lambda df, i: df['ATR_pct'].iloc[i] > 4 if i < len(df) else False,
    
    # 복합 조건
    "DC50/200 + 가격<MA50": lambda df, i: (df['DC_50_200'].iloc[i] and df['below_MA50'].iloc[i]) if i < len(df) else False,
    "DC50/200 + RSI<50": lambda df, i: (df['DC_50_200'].iloc[i] and df['RSI'].iloc[i] < 50) if i < len(df) else False,
    "가격<MA200 + RSI<50": lambda df, i: (df['below_MA200'].iloc[i] and df['RSI'].iloc[i] < 50) if i < len(df) else False,
    "고점-15% + RSI<50": lambda df, i: (df['drawdown_60'].iloc[i] <= -15 and df['RSI'].iloc[i] < 50) if i < len(df) else False,
    "MACD<0 + RSI<50": lambda df, i: (df['MACD_negative'].iloc[i] and df['RSI'].iloc[i] < 50) if i < len(df) else False,
    "DC100/200 + MACD<0": lambda df, i: (df['DC_100_200'].iloc[i] and df['MACD_negative'].iloc[i]) if i < len(df) else False,
    
    # 더 엄격한 복합
    "DC50/200 + 가격<MA200 + RSI<50": lambda df, i: (df['DC_50_200'].iloc[i] and df['below_MA200'].iloc[i] and df['RSI'].iloc[i] < 50) if i < len(df) else False,
    "고점-20% + MACD<0": lambda df, i: (df['drawdown_60'].iloc[i] <= -20 and df['MACD_negative'].iloc[i]) if i < len(df) else False,
}

# 분석 실행
print("\n[물타기 5회+ 거래 시작 시점에서 지표 감지율]\n")
print(f"{'지표':<35} | {'감지':>6} | {'총':>4} | {'감지율':>8} | {'평가':>8}")
print("-" * 75)

results = []
for name, condition in indicators.items():
    detected, total, rate = analyze_indicator_at_trade_start(heavy_trades, df, name, condition)
    
    if rate >= 70:
        grade = "🟢 우수"
    elif rate >= 50:
        grade = "🟡 양호"
    elif rate >= 30:
        grade = "🟠 보통"
    else:
        grade = "🔴 미흡"
    
    results.append((name, detected, total, rate, grade))
    print(f"{name:<35} | {detected:>5}회 | {total:>3} | {rate:>7.1f}% | {grade}")

# ===== 거짓 양성 (False Positive) 분석 =====
print("\n" + "=" * 100)
print("📊 거짓 양성 분석 (지표가 켜졌지만 물타기 적었던 경우)")
print("=" * 100)

# 물타기 4회 이하 거래
light_trades = [t for t in trades if t['num_buys'] < 5]

print("\n[물타기 4회 이하 거래에서 지표 오작동률]\n")
print(f"{'지표':<35} | {'오작동':>6} | {'총':>4} | {'오작동률':>8} | {'순감지율':>10}")
print("-" * 85)

for name, condition in indicators.items():
    # 물타기 많은 곳 감지율
    detected_heavy, total_heavy, rate_heavy = analyze_indicator_at_trade_start(heavy_trades, df, name, condition)
    # 물타기 적은 곳 오작동률
    detected_light, total_light, rate_light = analyze_indicator_at_trade_start(light_trades, df, name, condition)
    
    # 순 감지율 = 감지율 - 오작동률
    net_rate = rate_heavy - rate_light
    
    if net_rate >= 20:
        grade = "🟢 우수"
    elif net_rate >= 10:
        grade = "🟡 양호"
    elif net_rate >= 0:
        grade = "🟠 보통"
    else:
        grade = "🔴 역효과"
    
    print(f"{name:<35} | {detected_light:>5}회 | {total_light:>3} | {rate_light:>7.1f}% | {net_rate:>+8.1f}% {grade}")

# ===== 시기별 상세 분석 =====
print("\n" + "=" * 100)
print("📊 물타기 10회 이상 거래 시점의 지표 상태")
print("=" * 100)

very_heavy = [t for t in trades if t['num_buys'] >= 10]

for trade in sorted(very_heavy, key=lambda x: x['num_buys'], reverse=True):
    entry_date = trade['entry_dates'][0]
    idx = df.index.get_indexer([entry_date], method='ffill')[0]
    
    print(f"\n📍 {entry_date.strftime('%Y-%m-%d')} (물타기 {trade['num_buys']}회, {trade['exit_reason']} {trade['return']:+.1f}%)")
    
    if idx >= 0 and idx < len(df):
        row = df.iloc[idx]
        print(f"   가격: ${row['Close']:,.0f}")
        print(f"   RSI: {row['RSI']:.1f}")
        print(f"   DC 50/200: {'✓' if row['DC_50_200'] else '✗'} | DC 100/200: {'✓' if row['DC_100_200'] else '✗'}")
        print(f"   가격<MA50: {'✓' if row['below_MA50'] else '✗'} | <MA100: {'✓' if row['below_MA100'] else '✗'} | <MA200: {'✓' if row['below_MA200'] else '✗'}")
        print(f"   60일 고점대비: {row['drawdown_60']:.1f}%")
        print(f"   MACD: {row['MACD']:.0f} ({'음수' if row['MACD'] < 0 else '양수'})")
        print(f"   ATR%: {row['ATR_pct']:.2f}%")

print("\n" + "=" * 100)
print("✅ 분석 완료!")
print("=" * 100)

