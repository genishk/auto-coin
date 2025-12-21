"""
새 기준(20/50/75/45)을 4시간봉에 적용했을 때 비교
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators


def find_buy_signals(df, rsi_oversold, rsi_exit):
    signals = []
    in_oversold = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi < rsi_oversold and not in_oversold:
            in_oversold = True
        elif in_oversold and current_rsi >= rsi_exit:
            signals.append({
                'confirm_date': df.index[i],
                'confirm_price': row['Close']
            })
            in_oversold = False
    
    return signals


def find_sell_signals(df, rsi_overbought, rsi_exit):
    signals = []
    in_overbought = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_rsi = row.get('rsi', 50)
        
        if current_rsi > rsi_overbought and not in_overbought:
            in_overbought = True
        elif in_overbought and current_rsi <= rsi_exit:
            signals.append({
                'confirm_date': df.index[i],
                'confirm_price': row['Close']
            })
            in_overbought = False
    
    return signals


def simulate_new_strategy(df, buy_signals, sell_signals, stop_loss=-25):
    """새 전략: 수익일 때만 매도"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df.iloc[idx]['Close']
        
        if positions:
            avg_price = sum(p['price'] for p in positions) / len(positions)
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = current_price
            
            if current_return <= stop_loss:
                exit_reason = "손절"
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({'return': final_return})
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, positions


def test_params(df, oversold, buy_exit, overbought, sell_exit, label):
    """파라미터 테스트"""
    buy_signals = find_buy_signals(df, oversold, buy_exit)
    sell_signals = find_sell_signals(df, overbought, sell_exit)
    trades, positions = simulate_new_strategy(df, buy_signals, sell_signals, stop_loss=-25)
    
    if trades:
        wins = len([t for t in trades if t['return'] > 0])
        returns = [t['return'] for t in trades]
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r / 100)
        total_return = (cumulative - 1) * 100
        win_rate = wins / len(trades) * 100
    else:
        total_return = 0
        win_rate = 0
    
    print(f"\n{label}")
    print(f"   기준: 과매도<{oversold} → 탈출>={buy_exit}, 과매수>{overbought} → 탈출<={sell_exit}")
    print(f"   거래 횟수: {len(trades)}회")
    print(f"   승률: {win_rate:.0f}%")
    print(f"   총 수익률: {total_return:+.0f}%")
    print(f"   현재 물타기: {len(positions)}회")
    
    if positions:
        avg = sum(p['price'] for p in positions) / len(positions)
        print(f"   현재 평단가: ${avg:,.2f}")


def main():
    print("=" * 70)
    print("🔬 새 기준 vs 이전 기준 비교 (4시간봉 2년)")
    print("=" * 70)
    
    # 4시간봉 데이터
    print("\n📊 4시간봉 2년 데이터 로딩...")
    ticker = 'BTC-USD'
    cache = DataCache(cache_dir='data/cache_4h', max_age_hours=4)
    df_4h = cache.get(f'{ticker}_4h')
    
    if df_4h is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        df_4h = data[ticker]
        df_4h, _ = validate_data(df_4h, ticker)
        cache.set(f'{ticker}_4h', df_4h)
    
    ti = TechnicalIndicators()
    df_4h = ti.calculate_all(df_4h)
    
    print(f"   기간: {df_4h.index[0].strftime('%Y-%m-%d')} ~ {df_4h.index[-1].strftime('%Y-%m-%d')}")
    
    # 여러 기준 비교
    test_params(df_4h, 26, 30, 78, 30, "1️⃣ 이전 기준 (26/30/78/30) - 4시간봉")
    test_params(df_4h, 35, 40, 80, 55, "2️⃣ 평균수익1위 (35/40/80/55) - 4시간봉")
    test_params(df_4h, 30, 35, 70, 45, "3️⃣ 평균수익3위 (30/35/70/45) - 4시간봉")
    test_params(df_4h, 35, 55, 70, 55, "4️⃣ 물타기0회 (35/55/70/55) - 4시간봉")
    
    # 일봉 데이터
    print("\n" + "=" * 70)
    print("🔬 일봉 5년 데이터에서 비교")
    print("=" * 70)
    
    print("\n📊 일봉 5년 데이터 로딩...")
    cache_1d = DataCache(cache_dir='data/cache', max_age_hours=24)
    df_1d = cache_1d.get(f'{ticker}_1d')
    
    if df_1d is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='5y', interval='1d')
        df_1d = data[ticker]
        df_1d, _ = validate_data(df_1d, ticker)
        cache_1d.set(f'{ticker}_1d', df_1d)
    
    df_1d = ti.calculate_all(df_1d)
    
    print(f"   기간: {df_1d.index[0].strftime('%Y-%m-%d')} ~ {df_1d.index[-1].strftime('%Y-%m-%d')}")
    
    test_params(df_1d, 26, 30, 78, 30, "5️⃣ 이전 기준 (26/30/78/30) - 일봉")
    test_params(df_1d, 35, 40, 80, 55, "6️⃣ 평균수익1위 (35/40/80/55) - 일봉")
    test_params(df_1d, 30, 35, 70, 45, "7️⃣ 평균수익3위 (30/35/70/45) - 일봉")
    test_params(df_1d, 35, 55, 70, 55, "8️⃣ 물타기0회 (35/55/70/55) - 일봉")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

