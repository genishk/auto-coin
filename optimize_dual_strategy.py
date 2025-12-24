"""
롱/숏 양방향 전략 최적화
- dashboard_4h_dual.py와 100% 동일한 시뮬레이션 로직 사용
- 숏 파라미터 및 물타기 횟수 최적화
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from itertools import product
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config


def load_data(ticker: str):
    """4시간봉 데이터 로드 및 지표 계산"""
    config = load_config()
    
    cache = DataCache(
        cache_dir=str(project_root / "data" / "cache_4h"),
        max_age_hours=1
    )
    
    cache_key = f"{ticker}_4h"
    df = cache.get(cache_key)
    if df is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        if ticker in data:
            df = data[ticker]
            df, _ = validate_data(df, ticker)
            cache.set(cache_key, df)
    
    if df is not None:
        ti = TechnicalIndicators(config.get('indicators', {}))
        df = ti.calculate_all(df)
        
        # 이동평균선
        df['MA40'] = df['Close'].rolling(window=40).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['golden_cross'] = df['MA40'] > df['MA200']
    
    return df


# ========== 대시보드와 동일한 시그널 함수들 ==========

def find_long_signals(df: pd.DataFrame, rsi_oversold: float = 35, rsi_exit: float = 40, use_golden_cross: bool = True):
    """롱 진입 시그널 (대시보드와 동일)"""
    signals = []
    
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    last_signal_rsi = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        golden_cross_ok = True
        if use_golden_cross and 'golden_cross' in df.columns:
            gc = df['golden_cross'].iloc[idx]
            golden_cross_ok = gc if not pd.isna(gc) else False
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
            last_signal_rsi = rsi
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                if golden_cross_ok:
                    signals.append({
                        'type': 'long',
                        'signal_date': last_signal_date,
                        'signal_price': last_signal_price,
                        'signal_rsi': last_signal_rsi,
                        'confirm_date': df.index[idx],
                        'confirm_price': df['Close'].iloc[idx],
                        'confirm_rsi': rsi,
                        'golden_cross': golden_cross_ok
                    })
                in_oversold = False
                last_signal_date = None
    
    return signals


def find_long_exit_signals(df: pd.DataFrame, rsi_overbought: float = 80, rsi_exit: float = 55):
    """롱 청산 시그널 (대시보드와 동일)"""
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
            if in_overbought and rsi <= rsi_exit and last_signal_date is not None:
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


def find_short_signals(df: pd.DataFrame, rsi_peak: float = 80, rsi_exit: float = 70, lookback: int = 30):
    """숏 진입 시그널 (대시보드와 동일)"""
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
                    'type': 'short',
                    'signal_date': df.index[peak_idx],
                    'signal_price': df['Close'].iloc[peak_idx],
                    'signal_rsi': df['rsi'].iloc[peak_idx],
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': curr_rsi
                })
    
    return signals


def find_short_exit_signals(df: pd.DataFrame, rsi_oversold: float = 35, rsi_exit: float = 40):
    """숏 청산 시그널 (대시보드와 동일)"""
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


def simulate_dual_trades(df: pd.DataFrame, 
                         long_signals: list, long_exit_signals: list,
                         short_signals: list, short_exit_signals: list,
                         long_stop_loss: float = -25, short_stop_loss: float = -15,
                         short_max_hold: int = 60, short_max_entries: int = 2):
    """
    롱/숏 양방향 시뮬레이션 (대시보드와 동일 + 물타기 횟수 파라미터화)
    
    규칙:
    - 롱/숏 동시 보유 불가
    - 롱: 물타기 무제한, 수익시만 익절, 손절 long_stop_loss
    - 숏: 물타기 short_max_entries-1회까지, 수익시만 익절, 손절 short_stop_loss, 최대 보유 short_max_hold봉
    """
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
        
        # ===== 포지션 청산 체크 =====
        if positions and current_position:
            total_cost = sum(p['price'] for p in positions)
            avg_price = total_cost / len(positions)
            
            if current_position == 'long':
                current_return = (current_price / avg_price - 1) * 100
                stop_loss = long_stop_loss
            else:
                current_return = -((current_price / avg_price - 1) * 100)
                stop_loss = short_stop_loss
            
            exit_reason = None
            exit_price = current_price
            
            # 손절 체크
            if current_return <= stop_loss:
                exit_reason = "손절"
            
            # 익절 체크
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
            
            # 숏 최대 보유 기간 체크
            elif current_position == 'short' and entry_bar_idx is not None:
                bars_held = idx - entry_bar_idx
                if bars_held >= short_max_hold:
                    exit_reason = "기간만료"
            
            # 청산 실행
            if exit_reason:
                if current_position == 'long':
                    final_return = (exit_price / avg_price - 1) * 100
                else:
                    final_return = -((exit_price / avg_price - 1) * 100)
                
                trades.append({
                    'type': current_position,
                    'entry_dates': [p['date'] for p in positions],
                    'entry_prices': [p['price'] for p in positions],
                    'avg_price': avg_price,
                    'num_entries': len(positions),
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                
                current_position = None
                positions = []
                entry_bar_idx = None
        
        # ===== 신규 진입 체크 =====
        if current_position is None:
            if current_date in long_entry_dates:
                current_position = 'long'
                positions.append({
                    'date': current_date,
                    'price': long_entry_dates[current_date]['confirm_price']
                })
                entry_bar_idx = idx
            
            elif current_date in short_entry_dates:
                current_position = 'short'
                positions.append({
                    'date': current_date,
                    'price': short_entry_dates[current_date]['confirm_price']
                })
                entry_bar_idx = idx
        
        # ===== 물타기 체크 =====
        elif current_position == 'long' and current_date in long_entry_dates:
            # 롱 물타기 (무제한)
            positions.append({
                'date': current_date,
                'price': long_entry_dates[current_date]['confirm_price']
            })
        
        elif current_position == 'short' and current_date in short_entry_dates:
            # 숏 물타기 (short_max_entries까지)
            if len(positions) < short_max_entries:
                positions.append({
                    'date': current_date,
                    'price': short_entry_dates[current_date]['confirm_price']
                })
    
    return trades


def calculate_metrics(trades: list):
    """거래 결과로부터 성과 지표 계산"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'long_trades': 0,
            'long_win_rate': 0,
            'long_total': 0,
            'short_trades': 0,
            'short_win_rate': 0,
            'short_total': 0,
            'stop_loss_count': 0,
            'expired_count': 0
        }
    
    long_trades = [t for t in trades if t['type'] == 'long']
    short_trades = [t for t in trades if t['type'] == 'short']
    
    total_trades = len(trades)
    wins = len([t for t in trades if t['return'] > 0])
    total_return = sum(t['return'] for t in trades)
    
    long_wins = len([t for t in long_trades if t['return'] > 0]) if long_trades else 0
    long_total = sum(t['return'] for t in long_trades) if long_trades else 0
    
    short_wins = len([t for t in short_trades if t['return'] > 0]) if short_trades else 0
    short_total = sum(t['return'] for t in short_trades) if short_trades else 0
    
    stop_loss_count = len([t for t in trades if t['exit_reason'] == '손절'])
    expired_count = len([t for t in trades if t['exit_reason'] == '기간만료'])
    
    return {
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100 if total_trades else 0,
        'avg_return': total_return / total_trades if total_trades else 0,
        'total_return': total_return,
        'long_trades': len(long_trades),
        'long_win_rate': long_wins / len(long_trades) * 100 if long_trades else 0,
        'long_total': long_total,
        'short_trades': len(short_trades),
        'short_win_rate': short_wins / len(short_trades) * 100 if short_trades else 0,
        'short_total': short_total,
        'stop_loss_count': stop_loss_count,
        'expired_count': expired_count
    }


def optimize_dual_strategy(df: pd.DataFrame, 
                           long_params: dict,
                           short_param_ranges: dict,
                           progress_callback=None):
    """
    롱/숏 전략 최적화
    
    Args:
        df: 데이터프레임
        long_params: 고정된 롱 전략 파라미터
        short_param_ranges: 숏 전략 파라미터 범위 (리스트)
    
    Returns:
        results: 모든 조합의 결과
        best_result: 최적 결과
    """
    
    # 롱 시그널은 고정 (기존 최적화된 롱 전략)
    long_signals = find_long_signals(
        df, 
        long_params['rsi_oversold'],
        long_params['rsi_exit'],
        long_params['use_golden_cross']
    )
    long_exit_signals = find_long_exit_signals(
        df,
        long_params['rsi_overbought'],
        long_params['rsi_sell']
    )
    
    # 숏 파라미터 조합 생성
    param_keys = list(short_param_ranges.keys())
    param_values = [short_param_ranges[k] for k in param_keys]
    combinations = list(product(*param_values))
    
    print(f"총 {len(combinations)}개 조합 테스트 중...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_keys, combo))
        
        if progress_callback:
            progress_callback(i, len(combinations))
        
        # 숏 시그널 생성
        short_signals = find_short_signals(
            df,
            params['rsi_peak'],
            params['rsi_entry'],
            params['lookback']
        )
        short_exit_signals = find_short_exit_signals(
            df,
            long_params['rsi_oversold'],  # 숏 청산도 롱 과매도 기준 사용
            params['rsi_exit']
        )
        
        # 시뮬레이션
        trades = simulate_dual_trades(
            df,
            long_signals,
            long_exit_signals,
            short_signals,
            short_exit_signals,
            long_params['stop_loss'],
            params['stop_loss'],
            params['max_hold'],
            params['max_entries']
        )
        
        # 성과 계산
        metrics = calculate_metrics(trades)
        
        result = {
            **params,
            **metrics
        }
        results.append(result)
    
    # 결과 정렬 (누적 수익률 기준)
    results.sort(key=lambda x: x['total_return'], reverse=True)
    
    return results


def main():
    print("=" * 60)
    print("🔄 롱/숏 양방향 전략 최적화")
    print("=" * 60)
    
    # 데이터 로드
    print("\n📊 BTC-USD 4시간봉 데이터 로드 중...")
    df = load_data('BTC-USD')
    print(f"✅ {len(df)}개 봉 로드 완료")
    print(f"📅 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 롱 전략 파라미터 (기존 최적화된 값 - 고정)
    long_params = {
        'rsi_oversold': 35,
        'rsi_exit': 40,
        'rsi_overbought': 80,
        'rsi_sell': 55,
        'use_golden_cross': True,
        'stop_loss': -25
    }
    
    print("\n🟢 롱 전략 (고정)")
    print(f"   - RSI 과매도: {long_params['rsi_oversold']}")
    print(f"   - RSI 탈출: {long_params['rsi_exit']}")
    print(f"   - RSI 과매수: {long_params['rsi_overbought']}")
    print(f"   - RSI 매도: {long_params['rsi_sell']}")
    print(f"   - 골든크로스: {long_params['use_golden_cross']}")
    print(f"   - 손절: {long_params['stop_loss']}%")
    
    # 숏 전략 파라미터 범위 (최적화 대상)
    # 약 1,600개 조합
    short_param_ranges = {
        'rsi_peak': [75, 78, 82, 85],              # 4가지
        'rsi_entry': [65, 68, 72, 75],              # 4가지
        'rsi_exit': [35, 40, 45],                   # 3가지
        'lookback': [24, 30, 42],                   # 3가지
        'stop_loss': [-10, -15, -20],               # 3가지
        'max_hold': [42, 60, 90],                   # 3가지 (7일, 10일, 15일)
        'max_entries': [1, 2, 3, 4, 5]              # 5가지 (물타기 0회~4회)
    }
    
    total_combos = 1
    for v in short_param_ranges.values():
        total_combos *= len(v)
    
    print(f"\n🔴 숏 전략 최적화 범위")
    print(f"   - RSI 피크: {short_param_ranges['rsi_peak']}")
    print(f"   - RSI 진입: {short_param_ranges['rsi_entry']}")
    print(f"   - RSI 청산: {short_param_ranges['rsi_exit']}")
    print(f"   - Lookback: {short_param_ranges['lookback']}")
    print(f"   - 손절: {short_param_ranges['stop_loss']}")
    print(f"   - 최대보유: {short_param_ranges['max_hold']}")
    print(f"   - 물타기 최대: {short_param_ranges['max_entries']}")
    print(f"\n   📈 총 {total_combos:,}개 조합")
    
    # 최적화 실행
    print("\n" + "=" * 60)
    print("🚀 최적화 시작...")
    start_time = datetime.now()
    
    def progress(current, total):
        if current % 500 == 0:
            elapsed = (datetime.now() - start_time).seconds
            pct = current / total * 100
            print(f"   진행: {current:,}/{total:,} ({pct:.1f}%) - {elapsed}초 경과")
    
    results = optimize_dual_strategy(df, long_params, short_param_ranges, progress)
    
    elapsed = (datetime.now() - start_time).seconds
    print(f"\n✅ 완료! (소요 시간: {elapsed}초)")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 상위 20개 결과 (누적 수익률 순)")
    print("=" * 60)
    
    print(f"\n{'순위':>4} | {'RSI피크':>6} | {'진입':>4} | {'청산':>4} | {'LB':>3} | {'손절':>5} | {'보유':>4} | {'물타기':>4} | {'롱수익':>7} | {'숏수익':>7} | {'총수익':>8} | {'승률':>5}")
    print("-" * 110)
    
    for i, r in enumerate(results[:20]):
        print(f"{i+1:>4} | {r['rsi_peak']:>6} | {r['rsi_entry']:>4} | {r['rsi_exit']:>4} | {r['lookback']:>3} | {r['stop_loss']:>5}% | {r['max_hold']:>4} | {r['max_entries']-1:>4}회 | {r['long_total']:>+7.1f}% | {r['short_total']:>+7.1f}% | {r['total_return']:>+8.1f}% | {r['win_rate']:>5.1f}%")
    
    # 최적 결과 상세
    best = results[0]
    print("\n" + "=" * 60)
    print("🏆 최적 조합 상세")
    print("=" * 60)
    
    print(f"\n🔴 숏 최적 파라미터:")
    print(f"   - RSI 피크: {best['rsi_peak']}")
    print(f"   - RSI 진입: {best['rsi_entry']}")
    print(f"   - RSI 청산: {best['rsi_exit']}")
    print(f"   - Lookback: {best['lookback']}봉")
    print(f"   - 손절: {best['stop_loss']}%")
    print(f"   - 최대 보유: {best['max_hold']}봉 ({best['max_hold']/6:.1f}일)")
    print(f"   - 물타기: {best['max_entries']-1}회")
    
    print(f"\n📈 성과:")
    print(f"   - 총 거래: {best['total_trades']}회 (롱 {best['long_trades']}회 / 숏 {best['short_trades']}회)")
    print(f"   - 승률: {best['win_rate']:.1f}%")
    print(f"   - 롱 수익: {best['long_total']:+.1f}% (승률 {best['long_win_rate']:.1f}%)")
    print(f"   - 숏 수익: {best['short_total']:+.1f}% (승률 {best['short_win_rate']:.1f}%)")
    print(f"   - 총 누적 수익: {best['total_return']:+.1f}%")
    print(f"   - 손절 횟수: {best['stop_loss_count']}회")
    print(f"   - 기간만료: {best['expired_count']}회")
    
    # 물타기 횟수별 분석
    print("\n" + "=" * 60)
    print("📊 물타기 횟수별 최적 결과")
    print("=" * 60)
    
    for max_entries in short_param_ranges['max_entries']:
        filtered = [r for r in results if r['max_entries'] == max_entries]
        if filtered:
            best_in_group = filtered[0]
            print(f"\n물타기 {max_entries-1}회:")
            print(f"   최적: 피크{best_in_group['rsi_peak']}/진입{best_in_group['rsi_entry']}/청산{best_in_group['rsi_exit']}/손절{best_in_group['stop_loss']}%/보유{best_in_group['max_hold']}봉")
            print(f"   성과: 총 {best_in_group['total_return']:+.1f}% (롱 {best_in_group['long_total']:+.1f}% / 숏 {best_in_group['short_total']:+.1f}%)")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(project_root / "optimization_results.csv", index=False)
    print(f"\n📁 결과 저장: optimization_results.csv")
    
    return results


if __name__ == "__main__":
    main()

