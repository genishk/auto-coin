"""
GitHub Actions용 4시간봉 상세 시그널 체크 스크립트
- 현재 포지션 상태
- 물타기 횟수
- 숏 헷징 상태
- 실제 취해야 할 액션 명시
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

def find_sell_signals(df, rsi_overbought=80, rsi_exit=55):
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

def simulate_current_position(df, buy_signals, sell_signals, stop_loss=-25, 
                              hedge_threshold=2, hedge_upgrade_interval=3, 
                              hedge_profit=8, hedge_stop=-15):
    """현재 포지션 상태 시뮬레이션 (헷징 포함)"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    positions = []
    trades = []
    current_hedge = None
    hedge_trades = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        current_high = df['High'].iloc[idx]
        current_low = df['Low'].iloc[idx]
        macd_val = df['MACD'].iloc[idx] if 'MACD' in df.columns else 0
        
        # 숏 헷징 청산 체크
        if current_hedge is not None:
            target_price = current_hedge['entry_price'] * (1 - hedge_profit / 100)
            stop_price_hedge = current_hedge['entry_price'] * (1 - hedge_stop / 100)
            
            short_exit_reason = None
            if current_low <= target_price:
                short_exit_reason = f"숏익절+{hedge_profit}%"
            elif current_high >= stop_price_hedge:
                short_exit_reason = f"숏손절{hedge_stop}%"
            
            if short_exit_reason:
                hedge_trades.append({
                    'entry_date': current_hedge['entry_date'],
                    'exit_date': current_date,
                    'exit_reason': short_exit_reason,
                    'invested': current_hedge['invested']
                })
                current_hedge = None
        
        # 롱 포지션 처리
        if positions:
            total_qty = sum(1/p['price'] for p in positions)
            avg_price = len(positions) / total_qty
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            
            if current_return <= stop_loss:
                exit_reason = "손절"
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
            
            if exit_reason:
                trades.append({
                    'entry_dates': [p['date'] for p in positions],
                    'num_buys': len(positions),
                    'exit_date': current_date,
                    'exit_reason': exit_reason
                })
                # 롱 청산시 숏도 같이 청산
                if current_hedge is not None:
                    hedge_trades.append({
                        'entry_date': current_hedge['entry_date'],
                        'exit_date': current_date,
                        'exit_reason': '롱청산시',
                        'invested': current_hedge['invested']
                    })
                    current_hedge = None
                positions = []
        
        # 매수 처리
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
            
            num_buys = len(positions)
            
            # 헷징 진입/업그레이드 체크
            should_hedge = False
            if num_buys == hedge_threshold and current_hedge is None:
                should_hedge = True
            elif num_buys > hedge_threshold and hedge_upgrade_interval > 0:
                if (num_buys - hedge_threshold) % hedge_upgrade_interval == 0:
                    should_hedge = True
            
            if should_hedge and macd_val < 0:
                # 기존 숏 청산 (업그레이드)
                if current_hedge is not None:
                    hedge_trades.append({
                        'entry_date': current_hedge['entry_date'],
                        'exit_date': current_date,
                        'exit_reason': '업그레이드',
                        'invested': current_hedge['invested']
                    })
                
                # 새 숏 진입
                current_hedge = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'invested': num_buys * 1000
                }
    
    return positions, trades, current_hedge, hedge_trades

def main():
    ticker = 'BTC-USD'
    
    # 데이터 로드 (4시간봉, 2년)
    cache = DataCache(cache_dir='data/cache_4h', max_age_hours=4)
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
    sell_signals = find_sell_signals(df, 80, 55)
    
    # 현재 포지션 시뮬레이션 (헷징 포함)
    current_positions, trades, current_hedge, hedge_trades = simulate_current_position(
        df, buy_signals, sell_signals, -25,
        hedge_threshold=2, hedge_upgrade_interval=3, hedge_profit=8, hedge_stop=-15
    )
    
    # 시그널 체크 (최근 시점)
    buy_signal_today = False
    sell_signal_today = False
    
    if buy_signals and buy_signals[-1]['confirm_date'] == df.index[-1]:
        buy_signal_today = True
    if sell_signals and sell_signals[-1]['confirm_date'] == df.index[-1]:
        sell_signal_today = True
    
    # 현재 상태 계산
    has_position = len(current_positions) > 0
    water_count = len(current_positions) if has_position else 0
    
    if has_position:
        total_qty = sum(1/p['price'] for p in current_positions)
        avg_price = len(current_positions) / total_qty
        unrealized = (current_price / avg_price - 1) * 100
        invested = water_count * 1000  # $1,000 per entry
    else:
        avg_price = 0
        unrealized = 0
        invested = 0
    
    # 헷징 상태 판단
    hedge_threshold = 2
    hedge_upgrade_interval = 3
    should_hedge = False
    hedge_action = None
    
    if has_position and current_macd < 0:
        if water_count == hedge_threshold:
            should_hedge = True
            hedge_action = "첫 숏 헷징 진입"
        elif water_count > hedge_threshold:
            if (water_count - hedge_threshold) % hedge_upgrade_interval == 0:
                should_hedge = True
                hedge_action = f"숏 헷징 업그레이드 ({water_count}회차)"
    
    # 실제 액션 결정
    actions = []
    
    if buy_signal_today:
        if not has_position:
            actions.append("🟢 첫 롱 진입 ($1,000)")
        else:
            actions.append(f"🔵 물타기 {water_count + 1}회차 ($1,000 추가)")
            if should_hedge:
                hedge_amount = (water_count + 1) * 1000
                actions.append(f"🟣 {hedge_action} (${hedge_amount:,})")
    
    if sell_signal_today and has_position:
        if unrealized > 0:
            actions.append(f"🟡 롱 익절 (수익률: {unrealized:+.1f}%)")
            actions.append("🔚 숏 포지션도 함께 청산")
        else:
            actions.append(f"⏸️ 매도 시그널이지만 손해({unrealized:+.1f}%)라 보류")
    
    # 결과 출력
    print('=' * 60)
    print('₿ Auto-Coin 4시간봉 상세 리포트')
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
        print(f'   물타기: {water_count}회')
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
            print(f'   투자금: ${hedge_invested:,}')
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
    
    if actions:
        print('📋 취해야 할 액션:')
        for action in actions:
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
            f.write(f'current_price={current_price:,.2f}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'has_position={"yes" if has_position else "no"}\n')
            f.write(f'water_count={water_count}\n')
            f.write(f'unrealized={unrealized:+.1f}\n')
            
            # 시그널 있으면 메일 발송 (보류 포함)
            if buy_signal_today:
                f.write('signal_type=buy\n')
            elif sell_signal_today:
                f.write('signal_type=sell\n')
            else:
                f.write('signal_type=none\n')
            
            f.write(f'actions={" | ".join(actions) if actions else "없음"}\n')
            
            # 숏 헷징 상태도 추가
            if current_hedge:
                hedge_return = (current_hedge['entry_price'] - current_price) / current_hedge['entry_price'] * 100
                f.write(f'hedge_status=보유중 ({hedge_return:+.1f}%)\n')
            else:
                f.write('hedge_status=없음\n')

if __name__ == '__main__':
    main()

