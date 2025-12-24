"""
일봉 5년 데이터로 최종 비교: 기존 vs 개선 전략
- 2022년 하락장 포함
- 대시보드와 동일한 계산 방식
"""

import pandas as pd
import sys
sys.path.insert(0, '.')
from src.data.cache import DataCache
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 일봉 데이터 로드
cache = DataCache(cache_dir='data/cache', max_age_hours=24)
df = cache.get('BTC-USD_1d')

ti = TechnicalIndicators(load_config().get('indicators', {}))
df = ti.calculate_all(df)

print('=' * 80)
print('📊 일봉 5년 데이터 최종 비교')
print('=' * 80)
print(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"총 봉 수: {len(df)}")

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
SHORT_MAX_HOLD = 42  # 일봉 기준 42일
SHORT_LOOKBACK = 24
SHORT_MAX_ENTRIES = 4

DC_RSI_THRESHOLD = 65


def run_simulation(df, ma_short, ma_long, use_dc_short=False, label=""):
    """통합 시뮬레이션"""
    df = df.copy()
    df[f'MA{ma_short}'] = df['Close'].rolling(window=ma_short).mean()
    df[f'MA{ma_long}'] = df['Close'].rolling(window=ma_long).mean()
    df['golden_cross'] = df[f'MA{ma_short}'] > df[f'MA{ma_long}']
    df['dead_cross'] = df[f'MA{ma_short}'] < df[f'MA{ma_long}']
    
    # 롱 시그널
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
    
    # 롱 청산 시그널
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
    
    # 숏 시그널
    short_signals = []
    for idx in range(SHORT_LOOKBACK, len(df)):
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        is_golden = df['golden_cross'].iloc[idx]
        is_dead = df['dead_cross'].iloc[idx]
        
        if pd.isna(curr_rsi) or pd.isna(prev_rsi) or pd.isna(is_golden):
            continue
        
        if use_dc_short:
            if is_golden:
                recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
                had_peak = any(recent_rsi > SHORT_RSI_PEAK)
                if had_peak and prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
                    short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx], 'type': 'GC'})
            elif is_dead:
                if prev_rsi > DC_RSI_THRESHOLD and curr_rsi <= DC_RSI_THRESHOLD:
                    short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx], 'type': 'DC'})
        else:
            recent_rsi = df['rsi'].iloc[idx-SHORT_LOOKBACK:idx]
            had_peak = any(recent_rsi > SHORT_RSI_PEAK)
            if had_peak and prev_rsi > SHORT_RSI_ENTRY and curr_rsi <= SHORT_RSI_ENTRY:
                short_signals.append({'confirm_date': df.index[idx], 'confirm_price': df['Close'].iloc[idx], 'type': 'orig'})
    
    # 숏 청산 시그널
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
    
    # 시뮬레이션
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
                
                year = cd.year
                trades.append({'type': cp, 'return': fr, 'year': year, 'reason': er})
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
    
    # 연도별
    years = {}
    for t in trades:
        y = t['year']
        if y not in years:
            years[y] = {'total': 0, 'long': 0, 'short': 0}
        years[y]['total'] += t['return']
        if t['type'] == 'long':
            years[y]['long'] += t['return']
        else:
            years[y]['short'] += t['return']
    
    return {
        'total_trades': len(trades),
        'long_trades': len(lt),
        'short_trades': len(st),
        'total_return': total_return,
        'long_return': long_return,
        'short_return': short_return,
        'win_rate': win_rate,
        'years': years,
        'short_signals_count': len(short_signals)
    }


# ========================================
# 1. 기존 전략
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

print("\n연도별 수익:")
for y in sorted(result_orig['years'].keys()):
    yr = result_orig['years'][y]
    print(f"   {y}: 총 {yr['total']:+.1f}% (롱 {yr['long']:+.1f}%, 숏 {yr['short']:+.1f}%)")


# ========================================
# 2. 개선안 A (MA40/200 + DC숏)
# ========================================
print("\n" + "=" * 80)
print("2️⃣ 개선안 A (MA40/200 + DC숏)")
print("   - 롱: MA40/200 골든크로스 필터")
print("   - 숏: GC→RSI 78→65 / DC→RSI 65 하향")
print("=" * 80)

result_a = run_simulation(df, 40, 200, use_dc_short=True)
print(f"\n📊 전체: {result_a['total_trades']}회, 승률 {result_a['win_rate']:.1f}%, 누적 {result_a['total_return']:+.1f}%")
print(f"🟢 롱:   {result_a['long_trades']}회, 누적 {result_a['long_return']:+.1f}%")
print(f"🔴 숏:   {result_a['short_trades']}회, 누적 {result_a['short_return']:+.1f}%")
print(f"\n   기존 대비: {result_a['total_return'] - result_orig['total_return']:+.1f}%")

print("\n연도별 수익:")
for y in sorted(result_a['years'].keys()):
    yr = result_a['years'][y]
    orig_yr = result_orig['years'].get(y, {'total': 0})
    diff = yr['total'] - orig_yr.get('total', 0)
    print(f"   {y}: 총 {yr['total']:+.1f}% (롱 {yr['long']:+.1f}%, 숏 {yr['short']:+.1f}%) [기존대비 {diff:+.1f}%]")


# ========================================
# 3. 개선안 B (MA100/200 + DC숏)
# ========================================
print("\n" + "=" * 80)
print("3️⃣ 개선안 B (MA100/200 + DC숏)")
print("   - 롱: MA100/200 골든크로스 필터")
print("   - 숏: GC→RSI 78→65 / DC→RSI 65 하향")
print("=" * 80)

result_b = run_simulation(df, 100, 200, use_dc_short=True)
print(f"\n📊 전체: {result_b['total_trades']}회, 승률 {result_b['win_rate']:.1f}%, 누적 {result_b['total_return']:+.1f}%")
print(f"🟢 롱:   {result_b['long_trades']}회, 누적 {result_b['long_return']:+.1f}%")
print(f"🔴 숏:   {result_b['short_trades']}회, 누적 {result_b['short_return']:+.1f}%")
print(f"\n   기존 대비: {result_b['total_return'] - result_orig['total_return']:+.1f}%")

print("\n연도별 수익:")
for y in sorted(result_b['years'].keys()):
    yr = result_b['years'][y]
    orig_yr = result_orig['years'].get(y, {'total': 0})
    diff = yr['total'] - orig_yr.get('total', 0)
    print(f"   {y}: 총 {yr['total']:+.1f}% (롱 {yr['long']:+.1f}%, 숏 {yr['short']:+.1f}%) [기존대비 {diff:+.1f}%]")


# ========================================
# 4. MA100/200만 (DC숏 없음)
# ========================================
print("\n" + "=" * 80)
print("4️⃣ 비교: MA100/200 (DC숏 없음)")
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
print("📊 최종 비교표 (일봉 5년)")
print("=" * 80)

print(f"\n{'전략':>25} | {'총수익':>10} | {'롱':>10} | {'숏':>10} | {'기존대비':>10}")
print('-' * 75)
print(f"{'1. 기존 (MA40/200, DC숏X)':>25} | {result_orig['total_return']:>+9.1f}% | {result_orig['long_return']:>+9.1f}% | {result_orig['short_return']:>+9.1f}% | {'기준':>10}")
print(f"{'2. MA40/200 + DC숏':>25} | {result_a['total_return']:>+9.1f}% | {result_a['long_return']:>+9.1f}% | {result_a['short_return']:>+9.1f}% | {result_a['total_return'] - result_orig['total_return']:>+9.1f}%")
print(f"{'3. MA100/200 + DC숏':>25} | {result_b['total_return']:>+9.1f}% | {result_b['long_return']:>+9.1f}% | {result_b['short_return']:>+9.1f}% | {result_b['total_return'] - result_orig['total_return']:>+9.1f}%")
print(f"{'4. MA100/200, DC숏X':>25} | {result_c['total_return']:>+9.1f}% | {result_c['long_return']:>+9.1f}% | {result_c['short_return']:>+9.1f}% | {result_c['total_return'] - result_orig['total_return']:>+9.1f}%")


# 2022년 특별 분석
print("\n" + "=" * 80)
print("📉 2022년 하락장 성과 비교")
print("=" * 80)

print(f"\n{'전략':>25} | {'2022 총':>10} | {'2022 롱':>10} | {'2022 숏':>10}")
print('-' * 65)
for name, result in [('1. 기존', result_orig), ('2. MA40/200+DC숏', result_a), ('3. MA100/200+DC숏', result_b), ('4. MA100/200', result_c)]:
    yr = result['years'].get(2022, {'total': 0, 'long': 0, 'short': 0})
    print(f"{name:>25} | {yr['total']:>+9.1f}% | {yr['long']:>+9.1f}% | {yr['short']:>+9.1f}%")


# 최적 전략
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

