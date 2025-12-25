"""
상승장/하락장 판별 방법 체계적 테스트
목표: 가장 확실하게 상승장/하락장을 판별하는 지표 찾기

검증 방법:
1. 각 지표로 "상승장/하락장" 판별
2. 상승장에서 롱만 했을 때 수익률
3. 하락장에서 숏만 했을 때 수익률
4. 전체 수익률 (롱+숏) 계산
5. 판별 정확도 측정
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

# 대시보드 함수 import
from dashboard_4h import find_buy_signals, find_sell_signals, simulate_trades

print("=" * 120)
print("🔬 상승장/하락장 판별 방법 체계적 테스트")
print("=" * 120)

# 파라미터
RSI_OVERSOLD = 35
RSI_BUY_EXIT = 40
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 55
STOP_LOSS = -25

# 숏 파라미터 (dual에서 가져옴)
SHORT_RSI_PEAK = 78
SHORT_RSI_ENTRY = 65
SHORT_RSI_EXIT = 45
SHORT_STOP_LOSS = -15
SHORT_MAX_HOLD = 42
SHORT_LOOKBACK = 24


def add_indicators(df):
    """다양한 지표 추가"""
    # 추가 MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # MA 기울기 (최근 10봉 대비)
    df['MA50_slope'] = (df['MA50'] - df['MA50'].shift(10)) / df['MA50'].shift(10) * 100
    df['MA100_slope'] = (df['MA100'] - df['MA100'].shift(10)) / df['MA100'].shift(10) * 100
    df['MA200_slope'] = (df['MA200'] - df['MA200'].shift(10)) / df['MA200'].shift(10) * 100
    
    # 가격 모멘텀
    df['return_20'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
    df['return_50'] = (df['Close'] / df['Close'].shift(50) - 1) * 100
    df['return_100'] = (df['Close'] / df['Close'].shift(100) - 1) * 100
    
    # 고점/저점 대비 위치
    df['high_20'] = df['High'].rolling(window=20).max()
    df['low_20'] = df['Low'].rolling(window=20).min()
    df['high_50'] = df['High'].rolling(window=50).max()
    df['low_50'] = df['Low'].rolling(window=50).min()
    
    # 고점 대비 하락률
    df['drawdown_20'] = (df['Close'] / df['high_20'] - 1) * 100
    df['drawdown_50'] = (df['Close'] / df['high_50'] - 1) * 100
    
    # RSI 평균
    df['rsi_avg_10'] = df['rsi'].rolling(window=10).mean()
    df['rsi_avg_20'] = df['rsi'].rolling(window=20).mean()
    
    return df


# 판별 함수들
def is_bull_price_above_ma200(row):
    return row['Close'] > row['MA200'] if pd.notna(row['MA200']) else None

def is_bull_price_above_ma100(row):
    return row['Close'] > row['MA100'] if pd.notna(row['MA100']) else None

def is_bull_price_above_ma50(row):
    return row['Close'] > row['MA50'] if pd.notna(row['MA50']) else None

def is_bull_golden_cross_50_200(row):
    return row['MA50'] > row['MA200'] if pd.notna(row['MA200']) else None

def is_bull_golden_cross_100_200(row):
    return row['MA100'] > row['MA200'] if pd.notna(row['MA200']) else None

def is_bull_ma_aligned(row):
    """MA 정렬: MA20 > MA50 > MA100 > MA200"""
    if pd.isna(row['MA200']):
        return None
    return row['MA20'] > row['MA50'] > row['MA100'] > row['MA200']

def is_bull_ma50_slope_positive(row):
    return row['MA50_slope'] > 0 if pd.notna(row['MA50_slope']) else None

def is_bull_ma100_slope_positive(row):
    return row['MA100_slope'] > 0 if pd.notna(row['MA100_slope']) else None

def is_bull_ma200_slope_positive(row):
    return row['MA200_slope'] > 0 if pd.notna(row['MA200_slope']) else None

def is_bull_return_20_positive(row):
    return row['return_20'] > 0 if pd.notna(row['return_20']) else None

def is_bull_return_50_positive(row):
    return row['return_50'] > 0 if pd.notna(row['return_50']) else None

def is_bull_rsi_above_50(row):
    return row['rsi'] > 50 if pd.notna(row['rsi']) else None

def is_bull_rsi_avg_above_50(row):
    return row['rsi_avg_20'] > 50 if pd.notna(row['rsi_avg_20']) else None

def is_bull_drawdown_small(row):
    """고점 대비 -10% 이내면 상승장"""
    return row['drawdown_50'] > -10 if pd.notna(row['drawdown_50']) else None

def is_bull_drawdown_very_small(row):
    """고점 대비 -5% 이내면 상승장"""
    return row['drawdown_50'] > -5 if pd.notna(row['drawdown_50']) else None

def is_bull_combo_1(row):
    """복합: 가격 > MA200 AND MA50 기울기 양수"""
    if pd.isna(row['MA200']) or pd.isna(row['MA50_slope']):
        return None
    return row['Close'] > row['MA200'] and row['MA50_slope'] > 0

def is_bull_combo_2(row):
    """복합: 골든크로스 AND RSI > 50"""
    if pd.isna(row['MA200']) or pd.isna(row['rsi']):
        return None
    return row['MA100'] > row['MA200'] and row['rsi'] > 50

def is_bull_combo_3(row):
    """복합: 가격 > MA100 AND 50봉 수익률 양수"""
    if pd.isna(row['MA100']) or pd.isna(row['return_50']):
        return None
    return row['Close'] > row['MA100'] and row['return_50'] > 0


# 모든 판별 방법
INDICATORS = [
    ("가격 > MA200", is_bull_price_above_ma200),
    ("가격 > MA100", is_bull_price_above_ma100),
    ("가격 > MA50", is_bull_price_above_ma50),
    ("GC (MA50/200)", is_bull_golden_cross_50_200),
    ("GC (MA100/200)", is_bull_golden_cross_100_200),
    ("MA 정렬 (20>50>100>200)", is_bull_ma_aligned),
    ("MA50 기울기 양수", is_bull_ma50_slope_positive),
    ("MA100 기울기 양수", is_bull_ma100_slope_positive),
    ("MA200 기울기 양수", is_bull_ma200_slope_positive),
    ("20봉 수익률 양수", is_bull_return_20_positive),
    ("50봉 수익률 양수", is_bull_return_50_positive),
    ("RSI > 50", is_bull_rsi_above_50),
    ("RSI 평균(20) > 50", is_bull_rsi_avg_above_50),
    ("고점대비 -10% 이내", is_bull_drawdown_small),
    ("고점대비 -5% 이내", is_bull_drawdown_very_small),
    ("복합: 가격>MA200 + MA50기울기↑", is_bull_combo_1),
    ("복합: GC + RSI>50", is_bull_combo_2),
    ("복합: 가격>MA100 + 50봉수익↑", is_bull_combo_3),
]


def simulate_short(df, is_bull_func):
    """하락장에서만 숏"""
    from dashboard_4h_dual import find_short_signals, find_short_exit_signals
    
    short_signals = find_short_signals(df, SHORT_RSI_PEAK, SHORT_RSI_ENTRY, SHORT_LOOKBACK, 55)
    short_exit_signals = find_short_exit_signals(df, RSI_OVERSOLD, SHORT_RSI_EXIT)
    
    # 하락장일 때만 숏 시그널 필터링
    filtered_signals = []
    for sig in short_signals:
        idx = df.index.get_loc(sig['confirm_date'])
        row = df.iloc[idx]
        is_bull = is_bull_func(row)
        if is_bull is not None and not is_bull:  # 하락장
            filtered_signals.append(sig)
    
    # 간단한 숏 시뮬레이션
    short_entry_dates = {s['confirm_date']: s for s in filtered_signals}
    short_exit_dates = {s['confirm_date']: s for s in short_exit_signals}
    
    trades = []
    position = None
    entry_price = None
    entry_idx = None
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if position:
            current_return = -((current_price / entry_price - 1) * 100)
            
            exit_reason = None
            if current_return <= SHORT_STOP_LOSS:
                exit_reason = "손절"
            elif current_date in short_exit_dates and current_return > 0:
                exit_reason = "익절"
            elif idx - entry_idx >= SHORT_MAX_HOLD and current_return > 0:
                exit_reason = "기간만료"
            
            if exit_reason:
                trades.append({'return': current_return, 'reason': exit_reason})
                position = None
                entry_price = None
                entry_idx = None
        
        if not position and current_date in short_entry_dates:
            position = 'short'
            entry_price = short_entry_dates[current_date]['confirm_price']
            entry_idx = idx
    
    return trades


def simulate_long_filtered(df, is_bull_func):
    """상승장에서만 롱"""
    buy_signals = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, False)  # 골든크로스 필터 OFF
    sell_signals = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
    
    # 상승장일 때만 매수 시그널 필터링
    filtered_signals = []
    for sig in buy_signals:
        idx = df.index.get_loc(sig['confirm_date'])
        row = df.iloc[idx]
        is_bull = is_bull_func(row)
        if is_bull is not None and is_bull:  # 상승장
            filtered_signals.append(sig)
    
    # 시뮬레이션
    trades, _ = simulate_trades(df, filtered_signals, sell_signals, STOP_LOSS)
    return trades


def test_indicator(df, name, is_bull_func):
    """지표 테스트"""
    # 상승장/하락장 비율 계산
    bull_count = 0
    bear_count = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        result = is_bull_func(row)
        if result is True:
            bull_count += 1
        elif result is False:
            bear_count += 1
    
    total = bull_count + bear_count
    bull_ratio = bull_count / total * 100 if total > 0 else 0
    
    # 롱 테스트 (상승장에서만)
    long_trades = simulate_long_filtered(df, is_bull_func)
    long_return = sum(t['return'] for t in long_trades)
    long_count = len(long_trades)
    
    # 숏 테스트 (하락장에서만)
    short_trades = simulate_short(df, is_bull_func)
    short_return = sum(t['return'] for t in short_trades)
    short_count = len(short_trades)
    
    total_return = long_return + short_return
    
    return {
        'name': name,
        'bull_ratio': bull_ratio,
        'long_return': long_return,
        'long_count': long_count,
        'short_return': short_return,
        'short_count': short_count,
        'total_return': total_return
    }


# ===== 데이터 로드 =====
print("\n📊 데이터 로드...")
df = pd.read_csv("data/btc_4h_5y.csv", index_col=0, parse_dates=True)
df = add_indicators(df)
df = df.dropna(subset=['MA200'])  # MA200 이후 데이터만
print(f"   기간: {df.index[0]} ~ {df.index[-1]} ({len(df)}봉)")

# ===== 기준점: 롱 전용 =====
print("\n📌 기준점 (롱 전용, 필터 없음):")
all_buy = find_buy_signals(df, RSI_OVERSOLD, RSI_BUY_EXIT, False)
all_sell = find_sell_signals(df, RSI_OVERBOUGHT, RSI_SELL_EXIT)
all_trades, _ = simulate_trades(df, all_buy, all_sell, STOP_LOSS)
baseline_return = sum(t['return'] for t in all_trades)
print(f"   롱 전용: {len(all_trades)}회, +{baseline_return:.1f}%")

# ===== 테스트 =====
print("\n" + "=" * 120)
print("🔬 지표별 테스트 결과")
print("=" * 120)

results = []
for name, func in INDICATORS:
    result = test_indicator(df, name, func)
    results.append(result)

# 결과 정렬 (총수익률 높은 순)
results.sort(key=lambda x: x['total_return'], reverse=True)

print(f"\n{'지표':<30} | {'상승장%':>8} | {'롱':>10} | {'숏':>10} | {'합계':>10} | {'기준대비':>10}")
print("-" * 95)

for r in results:
    diff = r['total_return'] - baseline_return
    diff_str = f"{diff:+.1f}%" if diff != 0 else "기준"
    print(f"{r['name']:<30} | {r['bull_ratio']:>7.1f}% | {r['long_return']:>+9.1f}% | {r['short_return']:>+9.1f}% | {r['total_return']:>+9.1f}% | {diff_str:>10}")

# 상위 5개 상세 분석
print("\n" + "=" * 120)
print("🏆 상위 5개 지표 상세")
print("=" * 120)

for i, r in enumerate(results[:5], 1):
    print(f"\n{i}. {r['name']}")
    print(f"   상승장 비율: {r['bull_ratio']:.1f}%")
    print(f"   롱: {r['long_count']}회, {r['long_return']:+.1f}%")
    print(f"   숏: {r['short_count']}회, {r['short_return']:+.1f}%")
    print(f"   합계: {r['total_return']:+.1f}% (기준대비 {r['total_return'] - baseline_return:+.1f}%)")

print("\n" + "=" * 120)
print(f"📌 기준점 (롱 전용): +{baseline_return:.1f}%")
print("✅ 테스트 완료!")
print("=" * 120)

