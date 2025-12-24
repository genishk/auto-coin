"""
최종 비교: 기존 전략 vs 개선 전략
- 대시보드와 완전히 동일한 계산 방식
- 롱/숏 GC/DC 기준 통일
"""

import pandas as pd
import yfinance as yf
import sys
sys.path.insert(0, '.')
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 데이터 로드
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

print('=' * 80)
print('📊 최종 비교: 기존 vs 개선 (대시보드 동일 계산)')
print('=' * 80)
print(f"기간: {df.index[0]} ~ {df.index[-1]}")

# 파라미터 (대시보드와 동일)
LONG_RSI_OVERSOLD = 35
LONG_RSI_EXIT = 40
LONG_RSI_OVERBOUGHT = 80
LONG_RSI_SELL = 55
LONG_STOP_LOSS = -25

SHORT_RSI_PEAK = 78
SHORT_RSI_ENTRY = 65
SHORT_RSI_EXIT = 45
SHORT_STOP_LOSS = -15
SHORT_MAX_HOLD = 42
SHORT_LOOKBACK = 24
SHORT_MAX_ENTRIES = 4

DC_RSI_THRESHOLD = 65


def run_simulation(df, ma_short, ma_long, use_dc_short=False):
    """
    통합 시뮬레이션
    - ma_short/ma_long: GC/DC 판단 기준
    - use_dc_short: 데드크로스에서 RSI 65 하향 숏 사용 여부
    """
    # MA 계산
    df = df.copy()
    df[f'MA{ma_short}'] = df['Close'].rolling(window=ma_short).mean()
    df[f'MA{ma_long}'] = df['Close'].rolling(window=ma_long).mean()
    df['golden_cross'] = df[f'MA{ma_short}'] > df[f'MA{ma_long}']
    df['dead_cross'] = df[f'MA{ma_short}'] < df[f'MA{ma_long}']
    
    # === 롱 시그널 (GC 필터) ===
    long_signals = []
    in_oversold = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        gc = df['golden_cross'].iloc[idx]
        
        if pd.isna(rsi) or pd.isna(gc):
            continue
        
        if rsi < LONG_RSI_OVERSOLD:
            in_oversold = True
            last_date = df.index[idx]
        else:
            if in_oversold and rsi >= LONG_RSI_EXIT and last_date and gc:
                long_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
                in_oversold = False
                last_date = None
    
    # === 롱 청산 시그널 ===
    long_exit_signals = []
    in_ob = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        if rsi > LONG_RSI_OVERBOUGHT:
            in_ob = True
            last_date = df.index[idx]
        else:
            if in_ob and rsi <= LONG_RSI_SELL and last_date:
                long_exit_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
                in_ob = False
                last_date = None
    
    # === 숏 시그널 ===
    short_signals = []
    
    for idx in range(SHORT_LOOKBACK, len(df)):
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        is_golden = df['golden_cross'].iloc[idx]
        is_dead = df['dead_cross'].iloc[idx]
        
        if pd.isna(curr_rsi) or pd.isna(prev_rsi) or pd.isna(is_golden):
            continue
        
        if use_dc_short:
            # 개선된 전략: GC/DC 구분
            if is_golden:
                # 골든크로스: RSI peak 후 하락
                recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
                had_peak = any(recent_rsi > SHORT_RSI_PEAK)
                if had_peak and prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
                    short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
            elif is_dead:
                # 데드크로스: RSI threshold 하향
                if prev_rsi > DC_RSI_THRESHOLD and curr_rsi <= DC_RSI_THRESHOLD:
                    short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
        else:
            # 기존 전략: GC/DC 무관, RSI peak만
            recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
            had_peak = any(recent_rsi > SHORT_RSI_PEAK)
            if had_peak and prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
                short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
    
    # === 숏 청산 시그널 ===
    short_exit_signals = []
    in_os = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        if rsi < LONG_RSI_OVERSOLD:
            in_os = True
            last_date = df.index[idx]
        else:
            if in_os and rsi >= SHORT_RSI_EXIT and last_date:
                short_exit_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx]})
                in_os = False
                last_date = None
    
    # === 시뮬레이션 (대시보드와 동일) ===
    le = {s['confirm_date']: s for s in long_signals}
    lx = {s['confirm_date']: s for s in long_exit_signals}
    se = {s['confirm_date']: s for s in short_signals}
    sx = {s['confirm_date']: s for s in short_exit_signals}
    
    trades = []
    cp = None
    pos = []
    ebi = None
    
    for idx in range(ma_long, len(df)):
        cd = df.index[idx]
        cprice = df['Close'].iloc[idx]
        
        if pos and cp:
            tq = sum(1 / p['price'] for p in pos)
            ap = len(pos) / tq
            
            if cp == 'long':
                cr = (cprice / ap - 1) * 100
                sl = LONG_STOP_LOSS
            else:
                cr = -((cprice / ap - 1) * 100)
                sl = SHORT_STOP_LOSS
            
            er = None
            ep = cprice
            
            if cr <= sl:
                er = "손절"
            elif cp == 'long' and cd in lx:
                if cr > 0:
                    er = "익절"
                    ep = lx[cd]['confirm_price']
            elif cp == 'short' and cd in sx:
                epc = sx[cd]['confirm_price']
                ccr = -((epc / ap - 1) * 100)
                if ccr > 0:
                    er = "익절"
                    ep = epc
            elif cp == 'short' and ebi:
                bh = idx - ebi
                if bh >= SHORT_MAX_HOLD and cr > 0:
                    er = "기간만료"
            
            if er:
                if cp == 'long':
                    fr = (ep / ap - 1) * 100
                else:
                    fr = -((ep / ap - 1) * 100)
                trades.append({'type': cp, 'return': fr})
                cp = None
                pos = []
                ebi = None
        
        if cp is None:
            if cd in le:
                cp = 'long'
                pos = [{'date': cd, 'price': le[cd]['confirm_price']}]
                ebi = idx
            elif cd in se:
                cp = 'short'
                pos = [{'date': cd, 'price': se[cd]['confirm_price']}]
                ebi = idx
        elif cp == 'long' and cd in le:
            pos.append({'date': cd, 'price': le[cd]['confirm_price']})
        elif cp == 'short' and cd in se:
            if len(pos) < SHORT_MAX_ENTRIES:
                pos.append({'date': cd, 'price': se[cd]['confirm_price']})
    
    # 결과 계산
    lt = [t for t in trades if t['type'] == 'long']
    st = [t for t in trades if t['type'] == 'short']
    
    total_return = sum(t['return'] for t in trades)
    long_return = sum(t['return'] for t in lt)
    short_return = sum(t['return'] for t in st)
    
    wins = len([t for t in trades if t['return'] > 0])
    win_rate = wins / len(trades) * 100 if trades else 0
    
    return {
        'total_trades': len(trades),
        'long_trades': len(lt),
        'short_trades': len(st),
        'total_return': total_return,
        'long_return': long_return,
        'short_return': short_return,
        'win_rate': win_rate
    }


# ========================================
# 1. 기존 전략 (현재 대시보드: MA40/200, DC숏 없음)
# ========================================
print("\n" + "=" * 80)
print("1️⃣ 기존 전략 (현재 대시보드)")
print("   - 롱: MA40/200 골든크로스 필터")
print("   - 숏: RSI 78→65 (시장 상태 무관)")
print("=" * 80)

result_orig = run_simulation(df, 40, 200, use_dc_short=False)
print(f"\n📊 전체: {result_orig['total_trades']}회, 승률 {result_orig['win_rate']:.1f}%, 누적 {result_orig['total_return']:+.1f}%")
print(f"🟢 롱:   {result_orig['long_trades']}회, 누적 {result_orig['long_return']:+.1f}%")
print(f"🔴 숏:   {result_orig['short_trades']}회, 누적 {result_orig['short_return']:+.1f}%")


# ========================================
# 2. 개선안 A (MA40/200 유지 + DC숏 추가)
# ========================================
print("\n" + "=" * 80)
print("2️⃣ 개선안 A (MA40/200 + DC숏 추가)")
print("   - 롱: MA40/200 골든크로스 필터")
print("   - 숏: GC→RSI 78→65 / DC→RSI 65 하향")
print("=" * 80)

result_a = run_simulation(df, 40, 200, use_dc_short=True)
print(f"\n📊 전체: {result_a['total_trades']}회, 승률 {result_a['win_rate']:.1f}%, 누적 {result_a['total_return']:+.1f}%")
print(f"🟢 롱:   {result_a['long_trades']}회, 누적 {result_a['long_return']:+.1f}%")
print(f"🔴 숏:   {result_a['short_trades']}회, 누적 {result_a['short_return']:+.1f}%")
print(f"\n   기존 대비: {result_a['total_return'] - result_orig['total_return']:+.1f}%")


# ========================================
# 3. 개선안 B (MA100/200 + DC숏 추가)
# ========================================
print("\n" + "=" * 80)
print("3️⃣ 개선안 B (MA100/200 + DC숏 추가)")
print("   - 롱: MA100/200 골든크로스 필터")
print("   - 숏: GC→RSI 78→65 / DC→RSI 65 하향")
print("=" * 80)

result_b = run_simulation(df, 100, 200, use_dc_short=True)
print(f"\n📊 전체: {result_b['total_trades']}회, 승률 {result_b['win_rate']:.1f}%, 누적 {result_b['total_return']:+.1f}%")
print(f"🟢 롱:   {result_b['long_trades']}회, 누적 {result_b['long_return']:+.1f}%")
print(f"🔴 숏:   {result_b['short_trades']}회, 누적 {result_b['short_return']:+.1f}%")
print(f"\n   기존 대비: {result_b['total_return'] - result_orig['total_return']:+.1f}%")


# ========================================
# 4. 추가: MA100/200 + DC숏 없음 (비교용)
# ========================================
print("\n" + "=" * 80)
print("4️⃣ 비교: MA100/200 (DC숏 없음)")
print("   - 롱: MA100/200 골든크로스 필터")
print("   - 숏: RSI 78→65 (시장 상태 무관)")
print("=" * 80)

result_c = run_simulation(df, 100, 200, use_dc_short=False)
print(f"\n📊 전체: {result_c['total_trades']}회, 승률 {result_c['win_rate']:.1f}%, 누적 {result_c['total_return']:+.1f}%")
print(f"🟢 롱:   {result_c['long_trades']}회, 누적 {result_c['long_return']:+.1f}%")
print(f"🔴 숏:   {result_c['short_trades']}회, 누적 {result_c['short_return']:+.1f}%")
print(f"\n   기존 대비: {result_c['total_return'] - result_orig['total_return']:+.1f}%")


# ========================================
# 최종 비교표
# ========================================
print("\n" + "=" * 80)
print("📊 최종 비교표")
print("=" * 80)

print(f"\n{'전략':>25} | {'총수익':>10} | {'롱':>10} | {'숏':>10} | {'기존대비':>10}")
print('-' * 75)
print(f"{'1. 기존 (MA40/200, DC숏X)':>25} | {result_orig['total_return']:>+9.1f}% | {result_orig['long_return']:>+9.1f}% | {result_orig['short_return']:>+9.1f}% | {'기준':>10}")
print(f"{'2. MA40/200 + DC숏':>25} | {result_a['total_return']:>+9.1f}% | {result_a['long_return']:>+9.1f}% | {result_a['short_return']:>+9.1f}% | {result_a['total_return'] - result_orig['total_return']:>+9.1f}%")
print(f"{'3. MA100/200 + DC숏':>25} | {result_b['total_return']:>+9.1f}% | {result_b['long_return']:>+9.1f}% | {result_b['short_return']:>+9.1f}% | {result_b['total_return'] - result_orig['total_return']:>+9.1f}%")
print(f"{'4. MA100/200, DC숏X':>25} | {result_c['total_return']:>+9.1f}% | {result_c['long_return']:>+9.1f}% | {result_c['short_return']:>+9.1f}% | {result_c['total_return'] - result_orig['total_return']:>+9.1f}%")

# 최적 전략 판단
results = [
    ('기존', result_orig),
    ('MA40/200+DC숏', result_a),
    ('MA100/200+DC숏', result_b),
    ('MA100/200', result_c)
]
best = max(results, key=lambda x: x[1]['total_return'])

print(f"\n🏆 최적 전략: {best[0]}")
print(f"   총 수익: {best[1]['total_return']:+.1f}%")
print(f"   기존 대비: {best[1]['total_return'] - result_orig['total_return']:+.1f}%")

