"""
GitHub Actions용 ETH 4시간봉 상세 시그널 체크 스크립트
- 대시보드와 완전히 동일한 로직 사용
- 현재 포지션 상태
- 물타기 횟수
- 숏 헷징 상태 (50% 비율)
- 실제 취해야 할 액션 명시 (대시보드 타임라인과 동일)

ETH 최적 파라미터:
- RSI: 35/40 → 85/55
- 헷징: threshold=2, upgrade=5, ratio=50%, profit=8%, stop=-15%
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators
from datetime import datetime, timedelta
import os
import pandas as pd

# 대시보드와 동일한 시그널 함수
def find_buy_signals(df, rsi_oversold=35, rsi_exit=40, use_golden_cross=False):
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
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
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                if golden_cross_ok:
                    buy_signals.append({
                        'signal_date': last_signal_date,
                        'confirm_date': df.index[idx],
                        'confirm_price': df['Close'].iloc[idx]
                    })
                    in_oversold = False
                    last_signal_date = None
    
    return buy_signals

def find_sell_signals(df, rsi_overbought=85, rsi_exit=55):  # ETH 최적값: 85
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_signal_date = df.index[idx]
        else:
            if in_overbought and rsi <= rsi_exit and last_signal_date is not None:
                sell_signals.append({
                    'signal_date': last_signal_date,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df, buy_signals, sell_signals, stop_loss=-25, 
                   hedge_threshold=2, hedge_upgrade_interval=5, 
                   hedge_ratio=0.5, hedge_profit=8, hedge_stop=-15):
    """
    대시보드와 완전히 동일한 시뮬레이션 함수 (ETH 파라미터)
    추가: 각 날짜별 발생한 액션 리스트 반환
    """
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    hedge_trades = []
    current_hedge = None
    
    # 날짜별 액션 기록 (대시보드 타임라인과 동일)
    daily_actions = {}
    
    CAPITAL_PER_ENTRY = 1000
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        current_high = df['High'].iloc[idx]
        current_low = df['Low'].iloc[idx]
        macd_val = df['MACD'].iloc[idx] if 'MACD' in df.columns else 0
        
        today_actions = []
        
        # ===== 숏 헷징 청산 체크 =====
        if current_hedge is not None:
            short_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
            short_exit_reason = None
            short_exit_price = current_price
            
            target_price = current_hedge['entry_price'] * (1 - hedge_profit / 100)
            if current_low <= target_price:
                short_exit_reason = f"숏익절+{hedge_profit}%"
                short_exit_price = target_price
                short_return = hedge_profit
            
            stop_price = current_hedge['entry_price'] * (1 - hedge_stop / 100)
            if short_exit_reason is None and current_high >= stop_price:
                short_exit_reason = f"숏손절{hedge_stop}%"
                short_exit_price = stop_price
                short_return = hedge_stop
            
            if short_exit_reason:
                hedge_trades.append({
                    'entry_date': current_hedge['entry_date'],
                    'entry_price': current_hedge['entry_price'],
                    'exit_date': current_date,
                    'exit_price': short_exit_price,
                    'return': short_return,
                    'exit_reason': short_exit_reason,
                    'long_num_buys': current_hedge['long_num_buys'],
                    'invested': current_hedge.get('invested', current_hedge['long_num_buys'] * CAPITAL_PER_ENTRY * hedge_ratio)
                })
                
                # 액션 기록
                if "익절" in short_exit_reason:
                    today_actions.append(f"💰 {short_exit_reason} (${current_hedge['invested']:,.0f})")
                else:
                    today_actions.append(f"⛔ {short_exit_reason} (${current_hedge['invested']:,.0f})")
                
                current_hedge = None
        
        # ===== 롱 포지션 청산 체크 =====
        if positions:
            total_quantity = sum(1 / p['price'] for p in positions)
            avg_price = len(positions) / total_quantity
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
                trades.append({
                    'entry_dates': [p['date'] for p in positions],
                    'entry_prices': [p['price'] for p in positions],
                    'avg_price': avg_price,
                    'num_buys': len(positions),
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                
                # 롱 청산 액션 기록
                invested = len(positions) * CAPITAL_PER_ENTRY
                profit = invested * final_return / 100
                if exit_reason == "익절":
                    today_actions.append(f"🟡 롱 익절 ({final_return:+.1f}%, ${profit:+,.0f})")
                else:
                    today_actions.append(f"🔴 롱 손절 ({final_return:+.1f}%, ${profit:+,.0f})")
                
                # 롱 청산시 숏도 같이 청산
                if current_hedge is not None:
                    short_return = (current_hedge['entry_price'] - exit_price) / current_hedge['entry_price'] * 100
                    hedge_trades.append({
                        'entry_date': current_hedge['entry_date'],
                        'entry_price': current_hedge['entry_price'],
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return': short_return,
                        'exit_reason': '롱청산시',
                        'long_num_buys': current_hedge['long_num_buys'],
                        'invested': current_hedge.get('invested', current_hedge['long_num_buys'] * CAPITAL_PER_ENTRY * hedge_ratio)
                    })
                    today_actions.append(f"🔚 숏 롱청산시 청산 ({short_return:+.1f}%)")
                    current_hedge = None
                
                positions = []
        
        # ===== 매수 처리 =====
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
            
            num_buys = len(positions)
            invested = num_buys * CAPITAL_PER_ENTRY
            
            # 매수 액션 기록
            if num_buys == 1:
                today_actions.append(f"🟢 롱 첫 진입 (${CAPITAL_PER_ENTRY:,})")
            else:
                today_actions.append(f"🔵 물타기 {num_buys}회차 (${CAPITAL_PER_ENTRY:,} 추가, 총 ${invested:,})")
            
            # ===== 숏 헷징 진입/업그레이드 체크 =====
            should_hedge = False
            
            if num_buys == hedge_threshold and current_hedge is None:
                should_hedge = True
            elif num_buys > hedge_threshold and hedge_upgrade_interval > 0:
                if (num_buys - hedge_threshold) % hedge_upgrade_interval == 0:
                    should_hedge = True
            
            if should_hedge:
                if macd_val < 0:
                    # 기존 숏 청산 (업그레이드 시)
                    if current_hedge is not None:
                        short_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
                        hedge_trades.append({
                            'entry_date': current_hedge['entry_date'],
                            'entry_price': current_hedge['entry_price'],
                            'exit_date': current_date,
                            'exit_price': current_price,
                            'return': short_return,
                            'exit_reason': '업그레이드',
                            'long_num_buys': current_hedge['long_num_buys'],
                            'invested': current_hedge.get('invested', num_buys * CAPITAL_PER_ENTRY * hedge_ratio)
                        })
                        today_actions.append(f"🔄 숏 업그레이드 (기존 ${current_hedge['invested']:,.0f} 청산, {short_return:+.1f}%)")
                    
                    # 새 숏 진입 (롱 투자금 × 50% 비율)
                    long_invested = num_buys * CAPITAL_PER_ENTRY
                    short_invested = long_invested * hedge_ratio
                    
                    current_hedge = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'entry_idx': idx,
                        'long_num_buys': num_buys,
                        'invested': short_invested
                    }
                    today_actions.append(f"🟣 숏 헷징 진입 ({num_buys}회, ${short_invested:,.0f}, 50% 비율)")
                else:
                    # MACD >= 0 이라서 헷징 미발동
                    today_actions.append(f"⚪ 헷징 조건 도달했지만 MACD≥0 ({macd_val:.0f})이라 미발동")
        
        # ===== 매도 시그널 보류 체크 =====
        if current_date in all_sell_dates and positions:
            total_quantity = sum(1 / p['price'] for p in positions)
            avg_price = len(positions) / total_quantity
            sell_price = all_sell_dates[current_date]['confirm_price']
            sell_return = (sell_price / avg_price - 1) * 100
            
            if sell_return <= 0:
                today_actions.append(f"⏸️ 매도 시그널이지만 손해({sell_return:+.1f}%)라 보류")
        
        if today_actions:
            daily_actions[current_date] = today_actions
    
    return trades, positions, hedge_trades, current_hedge, daily_actions


def main():
    ticker = 'ETH-USD'
    
    # 데이터 로드 (4시간봉, 2년)
    cache = DataCache(cache_dir='data/cache_eth_4h', max_age_hours=4)
    df = cache.get(f'{ticker}_4h')
    
    if df is None:
        fetcher = CoinFetcher([ticker])
        data = fetcher.fetch(period='2y', interval='4h')
        df = data[ticker]
        df, _ = validate_data(df, ticker)
        cache.set(f'{ticker}_4h', df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators()
    df = ti.calculate_all(df)
    
    # 추가 지표
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # 최신 데이터
    latest = df.iloc[-1]
    current_time = df.index[-1].strftime('%Y-%m-%d %H:%M')
    current_rsi = latest.get('rsi', 0)
    current_price = latest['Close']
    current_macd = latest['MACD'] if 'MACD' in latest else 0
    
    # 시그널 생성 (대시보드와 동일)
    buy_signals = find_buy_signals(df, 35, 40, False)
    sell_signals = find_sell_signals(df, 85, 55)  # ETH: RSI 85
    
    # 시뮬레이션 (대시보드와 동일한 함수 사용)
    trades, current_positions, hedge_trades, current_hedge, daily_actions = simulate_trades(
        df, buy_signals, sell_signals, -25,
        hedge_threshold=2, hedge_upgrade_interval=5, 
        hedge_ratio=0.5, hedge_profit=8, hedge_stop=-15  # ETH 파라미터
    )
    
    # 시그널 체크 (최근 시점)
    buy_signal_today = False
    sell_signal_today = False
    
    if buy_signals and buy_signals[-1]['confirm_date'] == df.index[-1]:
        buy_signal_today = True
    if sell_signals and sell_signals[-1]['confirm_date'] == df.index[-1]:
        sell_signal_today = True
    
    # 오늘 발생한 액션 (대시보드 타임라인과 동일!)
    today_date = df.index[-1]
    today_actions = daily_actions.get(today_date, [])
    
    # 현재 상태 계산
    has_position = len(current_positions) > 0
    water_count = len(current_positions) if has_position else 0
    
    CAPITAL_PER_ENTRY = 1000
    
    if has_position:
        total_qty = sum(1/p['price'] for p in current_positions)
        avg_price = len(current_positions) / total_qty
        unrealized = (current_price / avg_price - 1) * 100
        invested = water_count * CAPITAL_PER_ENTRY
    else:
        avg_price = 0
        unrealized = 0
        invested = 0
    
    # 결과 출력
    print('=' * 60)
    print('💎 Auto-Coin ETH 4시간봉 상세 리포트')
    print('=' * 60)
    print()
    print(f'📅 시간: {current_time} (UTC)')
    print(f'💰 현재가: ${current_price:,.2f}')
    print(f'📊 RSI: {current_rsi:.1f}')
    print(f'📈 MACD: {current_macd:,.2f}')
    print()
    
    print('=' * 60)
    print('📍 현재 포지션 상태')
    print('=' * 60)
    
    if has_position:
        print(f'✅ 롱 포지션 보유 중')
        print(f'   매수 횟수: {water_count}회')
        print(f'   투자금: ${invested:,}')
        print(f'   평단가: ${avg_price:,.2f}')
        print(f'   미실현: {unrealized:+.1f}% (${invested * unrealized / 100:+,.0f})')
        
        # 손절 라인
        stop_price = avg_price * 0.75  # -25%
        print(f'   손절가: ${stop_price:,.2f} (-25%)')
        
        # 헷징 상태
        if current_macd < 0:
            print(f'   🛡️ MACD < 0: 헷징 조건 충족')
        else:
            print(f'   ⚪ MACD > 0: 헷징 대기')
        
        # 현재 숏 헷징 포지션
        if current_hedge:
            print()
            print(f'🟣 숏 헷징 포지션 보유 중')
            hedge_entry_price = current_hedge['entry_price']
            hedge_invested = current_hedge['invested']
            hedge_return = (hedge_entry_price - current_price) / hedge_entry_price * 100
            hedge_unrealized = hedge_invested * hedge_return / 100
            
            print(f'   진입가: ${hedge_entry_price:,.2f}')
            print(f'   투자금: ${hedge_invested:,.0f} (50% 비율)')
            print(f'   미실현: {hedge_return:+.1f}% (${hedge_unrealized:+,.0f})')
            
            # 익절/손절 라인
            target_price = hedge_entry_price * (1 - 8 / 100)
            stop_price_hedge = hedge_entry_price * (1 - (-15) / 100)
            print(f'   익절가: ${target_price:,.2f} (+8%)')
            print(f'   손절가: ${stop_price_hedge:,.2f} (-15%)')
        elif has_position and water_count >= 2:
            print()
            print(f'⚪ 숏 헷징 없음 (MACD >= 0 이었거나 조건 미충족)')
    else:
        print('❌ 포지션 없음 - 매수 시그널 대기')
    
    print()
    print('=' * 60)
    print('🚨 오늘의 시그널 & 액션')
    print('=' * 60)
    
    if buy_signal_today:
        print('🟢 매수 시그널 발생!')
    elif sell_signal_today:
        print('🔴 매도 시그널 발생!')
    else:
        print('📭 오늘 시그널 없음')
    
    print()
    
    if today_actions:
        print('📋 취해야 할 액션 (대시보드 타임라인과 동일):')
        for action in today_actions:
            print(f'   {action}')
    else:
        print('📋 현재 취할 액션 없음')
    
    print()
    print('=' * 60)
    
    # GitHub Actions 환경 변수
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f'current_time={current_time}\n')
            f.write(f'current_price=${current_price:,.2f}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'has_position={"yes" if has_position else "no"}\n')
            f.write(f'water_count={water_count}\n')
            f.write(f'unrealized={unrealized:+.1f}\n')
            
            # 시그널 있으면 메일 발송
            if buy_signal_today:
                f.write('signal_type=buy\n')
            elif sell_signal_today:
                f.write('signal_type=sell\n')
            else:
                f.write('signal_type=none\n')
            
            # 오늘 액션 (대시보드와 동일)
            f.write(f'actions={" | ".join(today_actions) if today_actions else "없음"}\n')
            
            # 숏 헷징 상태
            if current_hedge:
                hedge_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
                f.write(f'hedge_status=보유중 ({hedge_return:+.1f}%)\n')
            else:
                f.write('hedge_status=없음\n')

if __name__ == '__main__':
    main()
