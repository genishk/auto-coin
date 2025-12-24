"""
대시보드 계산 검증 스크립트
- 대시보드의 정확한 로직과 파라미터를 사용하여 계산
- 결과가 대시보드와 일치하는지 확인
"""

import pandas as pd
import yfinance as yf
import sys
sys.path.insert(0, '.')
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 대시보드와 동일하게 데이터 로드
ticker = yf.Ticker("BTC-USD")
df = ticker.history(period="2y", interval="1h")
df = df.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# 지표 계산
ti = TechnicalIndicators(load_config().get('indicators', {}))
df = ti.calculate_all(df)
df['MA40'] = df['Close'].rolling(window=40).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['golden_cross'] = df['MA40'] > df['MA200']

print('=' * 80)
print('🔍 대시보드 계산 검증')
print('=' * 80)
print(f"기간: {df.index[0]} ~ {df.index[-1]}")
print(f"총 봉 수: {len(df)}")

# ===== 대시보드와 동일한 파라미터 =====
long_rsi_oversold = 35
long_rsi_exit = 40
long_rsi_overbought = 80
long_rsi_sell = 55
long_stop_loss = -25
use_golden_cross = True

short_rsi_peak = 78
short_rsi_entry = 65
short_rsi_exit = 45
short_stop_loss = -15
short_max_hold = 42
short_lookback = 24
short_max_entries = 4  # 대시보드 기본값

print("\n📋 파라미터 (대시보드 기본값):")
print(f"롱: RSI {long_rsi_oversold}/{long_rsi_exit}/{long_rsi_overbought}/{long_rsi_sell}, 손절 {long_stop_loss}%")
print(f"숏: RSI peak {short_rsi_peak}, entry {short_rsi_entry}, exit {short_rsi_exit}, 손절 {short_stop_loss}%")
print(f"숏: lookback {short_lookback}, max_hold {short_max_hold}, max_entries {short_max_entries}")


# ===== 대시보드와 동일한 시그널 함수 =====
def find_long_signals(df, rsi_oversold=35, rsi_exit=40, use_gc=True):
    signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        gc = df['golden_cross'].iloc[idx] if use_gc else True
        
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None and gc:
                signals.append({
                    'type': 'long_entry',
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': rsi
                })
                in_oversold = False
                last_signal_date = None
    return signals


def find_long_exit_signals(df, rsi_overbought=80, rsi_sell=55):
    signals = []
    in_overbought = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_overbought and rsi <= rsi_sell and last_signal_date is not None:
                signals.append({
                    'type': 'long_exit',
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': rsi
                })
                in_overbought = False
                last_signal_date = None
    return signals


def find_short_signals(df, rsi_peak=80, rsi_exit=70, lookback=30):
    signals = []
    
    for idx in range(lookback, len(df)):
        recent_rsi = df['rsi'].iloc[idx-lookback:idx]
        had_peak = any(recent_rsi > rsi_peak)
        
        if not had_peak:
            continue
        
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        
        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            continue
        
        if prev_rsi > rsi_exit and curr_rsi <= rsi_exit:
            peak_idx = None
            for j in range(idx-1, max(idx-lookback, 0)-1, -1):
                if df['rsi'].iloc[j] > rsi_peak:
                    peak_idx = j
                    break
            
            if peak_idx is not None:
                signals.append({
                    'type': 'short_entry',
                    'peak_date': df.index[peak_idx],
                    'peak_price': df['Close'].iloc[peak_idx],
                    'peak_rsi': df['rsi'].iloc[peak_idx],
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': curr_rsi
                })
    return signals


def find_short_exit_signals(df, rsi_oversold=35, rsi_exit=45):
    signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                signals.append({
                    'type': 'short_exit',
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': rsi
                })
                in_oversold = False
                last_signal_date = None
    return signals


def simulate_dual_trades(df, long_signals, long_exit_signals, short_signals, short_exit_signals,
                         l_stop=-25, s_stop=-15, s_max_hold=42, s_max_entries=4):
    """대시보드와 동일한 시뮬레이션 로직"""
    long_entry_dates = {s['confirm_date']: s for s in long_signals}
    long_exit_dates = {s['confirm_date']: s for s in long_exit_signals}
    short_entry_dates = {s['confirm_date']: s for s in short_signals}
    short_exit_dates = {s['confirm_date']: s for s in short_exit_signals}
    
    trades = []
    current_position = None
    positions = []
    entry_bar_idx = None
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        # 포지션 청산 체크
        if positions and current_position:
            total_quantity = sum(1 / p['price'] for p in positions)
            avg_price = len(positions) / total_quantity
            
            if current_position == 'long':
                current_return = (current_price / avg_price - 1) * 100
                stop_loss = l_stop
            else:
                current_return = -((current_price / avg_price - 1) * 100)
                stop_loss = s_stop
            
            exit_reason = None
            exit_price = current_price
            
            if current_return <= stop_loss:
                exit_reason = "손절"
            elif current_position == 'long' and current_date in long_exit_dates:
                if current_return > 0:
                    exit_reason = "익절"
                    exit_price = long_exit_dates[current_date]['confirm_price']
            elif current_position == 'short' and current_date in short_exit_dates:
                exit_price_candidate = short_exit_dates[current_date]['confirm_price']
                candidate_return = -((exit_price_candidate / avg_price - 1) * 100)
                if candidate_return > 0:
                    exit_reason = "익절"
                    exit_price = exit_price_candidate
            elif current_position == 'short' and entry_bar_idx is not None:
                bars_held = idx - entry_bar_idx
                if bars_held >= s_max_hold and current_return > 0:
                    exit_reason = "기간만료"
            
            if exit_reason:
                if current_position == 'long':
                    final_return = (exit_price / avg_price - 1) * 100
                else:
                    final_return = -((exit_price / avg_price - 1) * 100)
                
                trades.append({
                    'type': current_position,
                    'entry_date': positions[0]['date'],
                    'num_entries': len(positions),
                    'avg_price': avg_price,
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                
                current_position = None
                positions = []
                entry_bar_idx = None
        
        # 신규 진입
        if current_position is None:
            if current_date in long_entry_dates:
                current_position = 'long'
                positions.append({'date': current_date, 'price': long_entry_dates[current_date]['confirm_price']})
                entry_bar_idx = idx
            elif current_date in short_entry_dates:
                current_position = 'short'
                positions.append({'date': current_date, 'price': short_entry_dates[current_date]['confirm_price']})
                entry_bar_idx = idx
        
        # 물타기
        elif current_position == 'long' and current_date in long_entry_dates:
            positions.append({'date': current_date, 'price': long_entry_dates[current_date]['confirm_price']})
        elif current_position == 'short' and current_date in short_entry_dates:
            if len(positions) < s_max_entries:
                positions.append({'date': current_date, 'price': short_entry_dates[current_date]['confirm_price']})
    
    return trades


# 시그널 계산
long_signals = find_long_signals(df, long_rsi_oversold, long_rsi_exit, use_golden_cross)
long_exit_signals = find_long_exit_signals(df, long_rsi_overbought, long_rsi_sell)
short_signals = find_short_signals(df, short_rsi_peak, short_rsi_entry, short_lookback)
short_exit_signals = find_short_exit_signals(df, long_rsi_oversold, short_rsi_exit)

print(f"\n📊 시그널 수:")
print(f"롱 진입: {len(long_signals)}, 롱 청산: {len(long_exit_signals)}")
print(f"숏 진입: {len(short_signals)}, 숏 청산: {len(short_exit_signals)}")

# 시뮬레이션
trades = simulate_dual_trades(df, long_signals, long_exit_signals, short_signals, short_exit_signals,
                              long_stop_loss, short_stop_loss, short_max_hold, short_max_entries)

# 결과 계산
long_trades = [t for t in trades if t['type'] == 'long']
short_trades = [t for t in trades if t['type'] == 'short']

total_trades = len(trades)
total_wins = len([t for t in trades if t['return'] > 0])
total_win_rate = total_wins / total_trades * 100 if total_trades else 0
total_avg_return = sum(t['return'] for t in trades) / total_trades if total_trades else 0
total_cumulative = sum(t['return'] for t in trades)

long_num = len(long_trades)
long_wins = len([t for t in long_trades if t['return'] > 0])
long_win_rate = long_wins / long_num * 100 if long_num else 0
long_avg = sum(t['return'] for t in long_trades) / long_num if long_num else 0
long_cumulative = sum(t['return'] for t in long_trades)

short_num = len(short_trades)
short_wins = len([t for t in short_trades if t['return'] > 0])
short_win_rate = short_wins / short_num * 100 if short_num else 0
short_avg = sum(t['return'] for t in short_trades) / short_num if short_num else 0
short_cumulative = sum(t['return'] for t in short_trades)

print("\n" + "=" * 80)
print("📊 계산 결과 (대시보드와 비교)")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────┐
│ 📊 전체 성과                                    │
├────────────────────────────────────────────────┤
│ 총 거래:     {total_trades:>6}회                          │
│ 승률:        {total_win_rate:>6.1f}%                         │
│ 평균 수익률: {total_avg_return:>+7.2f}%                        │
│ 누적 수익률: {total_cumulative:>+7.1f}%                        │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ 🟢 롱 성과                                      │
├────────────────────────────────────────────────┤
│ 롱 거래:     {long_num:>6}회                          │
│ 롱 승률:     {long_win_rate:>6.1f}%                         │
│ 롱 평균:     {long_avg:>+7.2f}%                        │
│ 롱 누적:     {long_cumulative:>+7.1f}%                        │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ 🔴 숏 성과                                      │
├────────────────────────────────────────────────┤
│ 숏 거래:     {short_num:>6}회                          │
│ 숏 승률:     {short_win_rate:>6.1f}%                         │
│ 숏 평균:     {short_avg:>+7.2f}%                        │
│ 숏 누적:     {short_cumulative:>+7.1f}%                        │
└────────────────────────────────────────────────┘
""")

print("\n" + "=" * 80)
print("🔍 대시보드 값과 비교:")
print("=" * 80)
print("""
대시보드 표시:
  총 거래: 43회, 승률: 97.7%, 평균: +4.08%, 누적: +175.3%
  롱: 23회, 95.7%, +5.44%, +125.1%
  숏: 20회, 100%, +2.51%, +50.3%
""")

# 차이점 출력
print("차이점 분석:")
if total_trades != 43:
    print(f"  ⚠️ 총 거래: {total_trades} vs 43 (차이: {total_trades - 43})")
if abs(total_cumulative - 175.3) > 1:
    print(f"  ⚠️ 누적 수익률: {total_cumulative:.1f}% vs 175.3% (차이: {total_cumulative - 175.3:.1f}%)")
if long_num != 23:
    print(f"  ⚠️ 롱 거래: {long_num} vs 23 (차이: {long_num - 23})")
if short_num != 20:
    print(f"  ⚠️ 숏 거래: {short_num} vs 20 (차이: {short_num - 20})")

# 거래 내역 출력
print("\n" + "=" * 80)
print("📋 전체 거래 내역")
print("=" * 80)
print(f"{'유형':>4} | {'진입일':>12} | {'청산일':>12} | {'물타기':>4} | {'평단가':>10} | {'청산가':>10} | {'수익률':>8} | {'사유':>6}")
print('-' * 90)
for t in trades:
    print(f"{'🟢롱' if t['type']=='long' else '🔴숏':>4} | {t['entry_date'].strftime('%Y-%m-%d'):>12} | {t['exit_date'].strftime('%Y-%m-%d'):>12} | {t['num_entries']:>4}회 | ${t['avg_price']:>8,.0f} | ${t['exit_price']:>8,.0f} | {t['return']:>+7.1f}% | {t['exit_reason']:>6}")

