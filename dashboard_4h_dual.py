"""
Auto-Coin 대시보드 (4시간봉) - 롱/숏 양방향 전략
streamlit run dashboard_4h_dual.py

4시간봉 기준 롱/숏 양방향 매매
- 롱: RSI 과매도 탈출 + 골든크로스 필터 (MA100/200)
- 숏: 골든크로스에서 RSI peak 후 하향, 데드크로스에서 RSI 55 하향
- 물타기 전략 시뮬레이션
- 하락장(2022) 방어 최적화 전략
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.cache import DataCache
from src.data.fetcher import CoinFetcher, validate_data
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 페이지 설정
st.set_page_config(
    page_title="Auto-Coin 롱/숏",
    page_icon="🔄",
    layout="wide"
)


@st.cache_data(ttl=900)
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
        
        # 이동평균선 (MA100/200 - 하락장 방어 최적화)
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['golden_cross'] = df['MA100'] > df['MA200']
        df['dead_cross'] = df['MA100'] < df['MA200']
    
    return df


def find_long_signals(df: pd.DataFrame, rsi_oversold: float = 35, rsi_exit: float = 40, use_golden_cross: bool = True):
    """
    롱 진입 시그널 찾기
    조건: RSI < rsi_oversold 후 → RSI >= rsi_exit 탈출 + 골든크로스
    """
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
    """
    롱 청산 시그널 찾기 (익절용)
    조건: RSI > rsi_overbought 후 → RSI <= rsi_exit 탈출
    """
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


def find_short_signals(df: pd.DataFrame, rsi_peak: float = 78, rsi_entry: float = 65, lookback: int = 24, dc_rsi_threshold: float = 55):
    """
    숏 진입 시그널 찾기 (하락장 방어 최적화)
    
    골든크로스(상승장):
        최근 lookback봉 내 RSI > rsi_peak 경험 + RSI <= rsi_entry 하락
    
    데드크로스(하락장):
        RSI > dc_rsi_threshold → RSI <= dc_rsi_threshold 하향
    """
    signals = []
    
    for idx in range(lookback, len(df)):
        curr_rsi = df['rsi'].iloc[idx]
        prev_rsi = df['rsi'].iloc[idx-1]
        
        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            continue
        
        is_gc = df['golden_cross'].iloc[idx] if 'golden_cross' in df.columns else True
        is_dc = df['dead_cross'].iloc[idx] if 'dead_cross' in df.columns else False
        
        # 골든크로스: RSI peak 전략
        if is_gc:
            recent_rsi = df['rsi'].iloc[idx-lookback:idx]
            had_peak = any(recent_rsi > rsi_peak)
            
            if had_peak and prev_rsi > rsi_entry and curr_rsi <= rsi_entry:
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
        
        # 데드크로스: RSI threshold 하향 전략 (하락장 방어)
        elif is_dc:
            if prev_rsi > dc_rsi_threshold and curr_rsi <= dc_rsi_threshold:
                signals.append({
                    'type': 'short',
                    'signal_date': df.index[idx],
                    'signal_price': df['Close'].iloc[idx],
                    'signal_rsi': prev_rsi,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': curr_rsi
                })
    
    return signals


def find_short_exit_signals(df: pd.DataFrame, rsi_oversold: float = 35, rsi_exit: float = 40):
    """
    숏 청산 시그널 찾기 (익절용)
    조건: RSI < rsi_oversold 후 → RSI >= rsi_exit 탈출
    """
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
                         short_max_hold: int = 42, short_max_entries: int = 4):
    """
    롱/숏 양방향 시뮬레이션
    
    규칙:
    - 롱/숏 동시 보유 불가 (한 번에 하나만)
    - 롱: 물타기 무제한, 수익시만 익절, 손절 -25%
    - 숏: 물타기 short_max_entries-1회, 수익시만 익절, 손절 -15%, 최대 보유 42봉(7일)
    """
    # 시그널 날짜별 인덱싱
    long_entry_dates = {s['confirm_date']: s for s in long_signals}
    long_exit_dates = {s['confirm_date']: s for s in long_exit_signals}
    short_entry_dates = {s['confirm_date']: s for s in short_signals}
    short_exit_dates = {s['confirm_date']: s for s in short_exit_signals}
    
    trades = []
    
    # 현재 포지션
    current_position = None  # 'long' or 'short' or None
    positions = []  # 포지션 리스트 (물타기용)
    entry_bar_idx = None  # 숏 최대 보유 기간 체크용
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        # ===== 포지션 청산 체크 =====
        if positions and current_position:
            # 동일 금액 투자 방식 평균가 계산
            # 매 진입마다 동일 금액 투자 → 저가에 더 많은 수량 구매
            total_quantity = sum(1 / p['price'] for p in positions)  # 1단위 금액당 수량 합계
            avg_price = len(positions) / total_quantity  # 총 금액 / 총 수량
            
            if current_position == 'long':
                current_return = (current_price / avg_price - 1) * 100
                stop_loss = long_stop_loss
            else:  # short
                current_return = -((current_price / avg_price - 1) * 100)
                stop_loss = short_stop_loss
            
            exit_reason = None
            exit_price = current_price
            
            # 손절 체크
            if current_return <= stop_loss:
                exit_reason = "손절"
            
            # 익절 체크 (수익일 때만)
            elif current_position == 'long' and current_date in long_exit_dates:
                if current_return > 0:
                    exit_reason = "익절"
                    exit_price = long_exit_dates[current_date]['confirm_price']
            
            elif current_position == 'short' and current_date in short_exit_dates:
                # 숏 익절: 현재 가격 기준으로 수익 체크
                exit_price_candidate = short_exit_dates[current_date]['confirm_price']
                candidate_return = -((exit_price_candidate / avg_price - 1) * 100)
                if candidate_return > 0:
                    exit_reason = "익절"
                    exit_price = exit_price_candidate
            
            # 숏 최대 보유 기간 체크 (profit_only 모드: 수익일 때만 청산)
            elif current_position == 'short' and entry_bar_idx is not None:
                bars_held = idx - entry_bar_idx
                if bars_held >= short_max_hold and current_return > 0:
                    exit_reason = "기간만료"
                # 손실이면 계속 보유 (익절 또는 손절까지 대기)
            
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
        # 포지션이 없을 때만 새 포지션 진입
        if current_position is None:
            # 롱 진입 체크
            if current_date in long_entry_dates:
                current_position = 'long'
                positions.append({
                    'date': current_date,
                    'price': long_entry_dates[current_date]['confirm_price']
                })
                entry_bar_idx = idx
            
            # 숏 진입 체크
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
    
    # 현재 보유 중인 포지션 정보
    current_positions_info = None
    if positions:
        # 동일 금액 투자 방식 평균가 계산
        total_quantity = sum(1 / p['price'] for p in positions)
        avg_price = len(positions) / total_quantity
        current_price = df['Close'].iloc[-1]
        
        if current_position == 'long':
            unrealized = (current_price / avg_price - 1) * 100
        else:
            unrealized = -((current_price / avg_price - 1) * 100)
        
        current_positions_info = {
            'type': current_position,
            'positions': positions,
            'avg_price': avg_price,
            'unrealized': unrealized,
            'bars_held': len(df) - 1 - entry_bar_idx if entry_bar_idx else 0
        }
    
    return trades, current_positions_info


def main():
    st.title("🔄 Auto-Coin 롱/숏 양방향 전략")
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # ===== 사이드바 설정 =====
    st.sidebar.header("⚙️ 설정")
    
    config = load_config()
    tickers = config.get('tickers', ['BTC-USD'])
    ticker = st.sidebar.selectbox("코인", tickers, index=0)
    
    lookback_days = st.sidebar.slider("차트 기간 (일)", 7, 730, 180)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🟢 롱 전략 설정")
    
    long_rsi_oversold = st.sidebar.slider("롱 과매도 기준", 20, 45, 35, key="long_oversold")
    long_rsi_exit = st.sidebar.slider("롱 매수 탈출", 30, 60, 40, key="long_exit")
    long_rsi_overbought = st.sidebar.slider("롱 과매수 기준", 70, 95, 80, key="long_overbought")
    long_rsi_sell = st.sidebar.slider("롱 매도 탈출", 40, 70, 55, key="long_sell")
    long_stop_loss = st.sidebar.slider("롱 손절 (%)", -40, -10, -25, key="long_sl")
    use_golden_cross = st.sidebar.checkbox("골든크로스 필터 (롱)", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 숏 전략 설정 (하락장 방어)")
    
    st.sidebar.caption("📈 상승장(GC): RSI peak 후 하향")
    short_rsi_peak = st.sidebar.slider("GC 숏 RSI 피크", 70, 90, 78, key="short_peak")
    short_rsi_entry = st.sidebar.slider("GC 숏 진입 RSI", 55, 80, 65, key="short_entry")
    short_lookback = st.sidebar.slider("GC RSI 피크 체크 (봉)", 18, 60, 24, key="short_lookback")
    
    st.sidebar.caption("📉 하락장(DC): RSI 하향 돌파")
    dc_rsi_threshold = st.sidebar.slider("DC 숏 진입 RSI", 45, 70, 55, key="dc_rsi")
    
    st.sidebar.caption("공통 설정")
    short_rsi_exit = st.sidebar.slider("숏 청산 RSI", 30, 55, 45, key="short_exit")
    short_stop_loss = st.sidebar.slider("숏 손절 (%)", -25, -5, -15, key="short_sl")
    short_max_hold = st.sidebar.slider("숏 최대 보유 (봉)", 30, 120, 42, key="short_hold")
    short_max_entries = st.sidebar.slider("숏 물타기 최대 횟수", 0, 5, 3, key="short_entries")
    
    # ===== 데이터 로드 =====
    with st.spinner(f"{ticker} 데이터 로딩 중..."):
        df = load_data(ticker)
    
    if df is None:
        st.error(f"❌ {ticker} 데이터를 불러올 수 없습니다.")
        return
    
    # 현재 상태
    current_gc = df['golden_cross'].iloc[-1] if 'golden_cross' in df.columns else False
    current_rsi = df['rsi'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    st.sidebar.success(f"✅ {len(df)}개 봉 로드")
    st.sidebar.info(f"📅 {df.index[0].date()} ~ {df.index[-1].date()}")
    
    if current_gc:
        st.sidebar.success("🟢 골든크로스 (롱 허용, GC숏)")
    else:
        st.sidebar.warning("🔴 데드크로스 (롱 제한, DC숏)")
    
    # ===== 시그널 계산 =====
    long_signals = find_long_signals(df, long_rsi_oversold, long_rsi_exit, use_golden_cross)
    long_exit_signals = find_long_exit_signals(df, long_rsi_overbought, long_rsi_sell)
    short_signals = find_short_signals(df, short_rsi_peak, short_rsi_entry, short_lookback, dc_rsi_threshold)
    short_exit_signals = find_short_exit_signals(df, long_rsi_oversold, short_rsi_exit)
    
    # ===== 시뮬레이션 =====
    trades, current_positions_info = simulate_dual_trades(
        df, long_signals, long_exit_signals, 
        short_signals, short_exit_signals,
        long_stop_loss, short_stop_loss, short_max_hold, short_max_entries + 1
    )
    
    # ===== 탭 구성 =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 현재 상태",
        "📈 롱/숏 시그널",
        "💹 전략 성과",
        "📉 거래 내역",
        "🔍 데이터 확인"
    ])
    
    # ===== 탭 1: 현재 상태 =====
    with tab1:
        st.header(f"📊 {ticker} 현재 상태")
        
        prev = df['Close'].iloc[-2]
        change = (current_price / prev - 1) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("현재가", f"${current_price:,.2f}", f"{change:+.2f}%")
        with col2:
            rsi_status = "🔴 과매도" if current_rsi < long_rsi_oversold else ("🟢 과매수" if current_rsi > long_rsi_overbought else "⚪ 중립")
            st.metric("RSI", f"{current_rsi:.1f}", delta=rsi_status)
        with col3:
            gc_status = "🟢 상승장" if current_gc else "🔴 하락장"
            st.metric("추세 (MA100/200)", gc_status)
        with col4:
            if current_positions_info:
                pos_type = "🟢 롱" if current_positions_info['type'] == 'long' else "🔴 숏"
                st.metric("현재 포지션", pos_type, 
                         delta=f"{current_positions_info['unrealized']:+.1f}%")
            else:
                st.metric("현재 포지션", "⏳ 대기 중")
        with col5:
            if trades:
                long_trades = [t for t in trades if t['type'] == 'long']
                short_trades = [t for t in trades if t['type'] == 'short']
                st.metric("거래 수", f"롱 {len(long_trades)} / 숏 {len(short_trades)}")
        
        st.divider()
        
        # ===== 현재 포지션 상세 =====
        if current_positions_info:
            pos_type = current_positions_info['type']
            pos_emoji = "🟢" if pos_type == 'long' else "🔴"
            pos_name = "롱" if pos_type == 'long' else "숏"
            
            st.subheader(f"{pos_emoji} 현재 {pos_name} 포지션")
            
            avg_price = current_positions_info['avg_price']
            unrealized = current_positions_info['unrealized']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균 진입가", f"${avg_price:,.2f}")
            with col2:
                st.metric("진입 횟수", f"{len(current_positions_info['positions'])}회")
            with col3:
                color = "🟢" if unrealized >= 0 else "🔴"
                st.metric("미실현 손익", f"{color} {unrealized:+.1f}%")
            with col4:
                bars_held = current_positions_info['bars_held']
                days_held = bars_held / 6
                st.metric("보유 기간", f"{bars_held}봉 ({days_held:.1f}일)")
            
            # 포지션 상세
            st.markdown("**📋 진입 내역**")
            pos_df = pd.DataFrame([{
                '진입일': p['date'].strftime('%Y-%m-%d %H:%M'),
                '진입가': f"${p['price']:,.2f}",
                '현재 손익': f"{((current_price/p['price']-1)*100 if pos_type=='long' else -((current_price/p['price']-1)*100)):+.1f}%"
            } for p in current_positions_info['positions']])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            
            # 청산 조건 안내
            if pos_type == 'long':
                st.info(f"""
                **📤 롱 청산 조건:**
                - RSI > {long_rsi_overbought} 발생 후 → RSI ≤ {long_rsi_sell} 탈출 + **수익시만** 익절
                - 손절: {long_stop_loss}% (현재: {unrealized:+.1f}%)
                """)
            else:
                remaining_bars = short_max_hold - bars_held
                st.info(f"""
                **📤 숏 청산 조건:**
                - RSI < {long_rsi_oversold} 발생 후 → RSI ≥ {short_rsi_exit} 탈출 + **수익시만** 익절
                - 손절: {short_stop_loss}% (현재: {unrealized:+.1f}%)
                - 최대 보유: {short_max_hold}봉 (남은: {remaining_bars}봉)
                """)
        else:
            st.subheader("⏳ 대기 중")
            st.info("현재 보유 포지션이 없습니다. 시그널 대기 중...")
            
            # 다음 시그널 예상
            if current_rsi < long_rsi_oversold and current_gc:
                st.warning(f"⚠️ RSI {current_rsi:.1f} - 롱 진입 구간! (탈출 대기: RSI ≥ {long_rsi_exit})")
            elif current_gc and current_rsi > short_rsi_peak:
                st.warning(f"⚠️ RSI {current_rsi:.1f} - GC숏 피크 감지! (진입 대기: RSI ≤ {short_rsi_entry})")
            elif not current_gc and current_rsi > dc_rsi_threshold:
                st.warning(f"⚠️ RSI {current_rsi:.1f} - DC숏 진입 대기! (RSI ≤ {dc_rsi_threshold})")
        
        st.divider()
        
        # ===== 가격 차트 =====
        st.subheader("📉 가격 차트")
        
        signal_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        chart_df = df[df.index >= signal_cutoff]
        filtered_trades = [t for t in trades if t['exit_date'] >= signal_cutoff]
        
        fig = go.Figure()
        
        # 캔들스틱
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='가격'
        ))
        
        # MA 라인
        if 'MA100' in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df['MA100'],
                mode='lines', line=dict(color='orange', width=1.5),
                name='MA100'
            ))
        if 'MA200' in chart_df.columns:
            fig.add_trace(go.Scatter(
                x=chart_df.index, y=chart_df['MA200'],
                mode='lines', line=dict(color='purple', width=1.5),
                name='MA200'
            ))
        
        # 거래 표시
        for trade in filtered_trades:
            is_long = trade['type'] == 'long'
            entry_color = 'limegreen' if is_long else 'red'
            exit_color = 'dodgerblue' if trade['return'] > 0 else 'crimson'
            
            # 첫 진입
            fig.add_trace(go.Scatter(
                x=[trade['entry_dates'][0]],
                y=[trade['entry_prices'][0]],
                mode='markers',
                marker=dict(color=entry_color, size=12, 
                           symbol='triangle-up' if is_long else 'triangle-down',
                           line=dict(width=1, color='black')),
                showlegend=False,
                hovertemplate=f"{'🟢 롱' if is_long else '🔴 숏'} 진입<br>${trade['entry_prices'][0]:,.2f}<extra></extra>"
            ))
            
            # 물타기
            for i in range(1, len(trade['entry_dates'])):
                fig.add_trace(go.Scatter(
                    x=[trade['entry_dates'][i]],
                    y=[trade['entry_prices'][i]],
                    mode='markers',
                    marker=dict(color=entry_color, size=10, symbol='diamond',
                               opacity=0.95, line=dict(width=2, color='white')),
                    showlegend=False,
                    hovertemplate=f"💧 물타기 {i}회<br>${trade['entry_prices'][i]:,.2f}<extra></extra>"
                ))
            
            # 청산
            fig.add_trace(go.Scatter(
                x=[trade['exit_date']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=exit_color, size=12, symbol='x',
                           line=dict(width=2)),
                showlegend=False,
                hovertemplate=f"{trade['exit_reason']}<br>${trade['exit_price']:,.2f}<br>{trade['return']:+.1f}%<extra></extra>"
            ))
        
        # 현재 포지션 표시
        if current_positions_info:
            for pos in current_positions_info['positions']:
                if pos['date'] >= signal_cutoff:
                    fig.add_trace(go.Scatter(
                        x=[pos['date']],
                        y=[pos['price']],
                        mode='markers',
                        marker=dict(color='gold', size=14, symbol='star',
                                   line=dict(width=2, color='orange')),
                        showlegend=False,
                        hovertemplate=f"보유중<br>${pos['price']:,.2f}<extra></extra>"
                    ))
        
        # 범례
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=10, symbol='triangle-up'), name='🟢 롱 진입'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'), name='🔴 숏 진입'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=10, symbol='diamond', line=dict(width=2, color='white')), name='💧 롱 물타기'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='red', size=10, symbol='diamond', line=dict(width=2, color='white')), name='💧 숏 물타기'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='dodgerblue', size=10, symbol='x'), name='🔵 익절'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='crimson', size=10, symbol='x'), name='🔴 손절'))
        
        fig.update_layout(
            height=550,
            xaxis_rangeslider_visible=False,
            title=f"가격 차트 (최근 {lookback_days}일)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== 탭 2: 시그널 분석 =====
    with tab2:
        st.header("📈 롱/숏 시그널 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 롱 시그널")
            st.caption(f"조건: RSI < {long_rsi_oversold} → RSI ≥ {long_rsi_exit} + {'골든크로스' if use_golden_cross else '필터없음'}")
            
            signal_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
            filtered_long = [s for s in long_signals if s['confirm_date'] >= signal_cutoff]
            
            if filtered_long:
                long_df = pd.DataFrame([{
                    '진입일': s['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                    '진입가': f"${s['confirm_price']:,.2f}",
                    'RSI': f"{s['confirm_rsi']:.1f}",
                } for s in sorted(filtered_long, key=lambda x: x['confirm_date'], reverse=True)[:10]])
                st.dataframe(long_df, use_container_width=True, hide_index=True)
            else:
                st.info("최근 롱 시그널 없음")
            
            st.metric("총 롱 시그널", f"{len(long_signals)}회")
        
        with col2:
            st.subheader("🔴 숏 시그널")
            st.caption(f"GC: RSI>{short_rsi_peak}→{short_rsi_entry} | DC: RSI→{dc_rsi_threshold} 하향")
            
            filtered_short = [s for s in short_signals if s['confirm_date'] >= signal_cutoff]
            
            if filtered_short:
                short_df = pd.DataFrame([{
                    '진입일': s['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                    '진입가': f"${s['confirm_price']:,.2f}",
                    'RSI': f"{s['confirm_rsi']:.1f}",
                } for s in sorted(filtered_short, key=lambda x: x['confirm_date'], reverse=True)[:10]])
                st.dataframe(short_df, use_container_width=True, hide_index=True)
            else:
                st.info("최근 숏 시그널 없음")
            
            st.metric("총 숏 시그널", f"{len(short_signals)}회")
        
        st.divider()
        
        # 통합 시그널 차트
        st.subheader("🎯 통합 시그널 차트")
        
        fig_signals = go.Figure()
        
        fig_signals.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='가격', line=dict(color='gray', width=1.5)
        ))
        
        # 롱 시그널
        fig_signals.add_trace(go.Scatter(
            x=[s['confirm_date'] for s in long_signals],
            y=[s['confirm_price'] for s in long_signals],
            mode='markers', name=f'🟢 롱 ({len(long_signals)}회)',
            marker=dict(color='limegreen', size=8, symbol='triangle-up')
        ))
        
        # 숏 시그널
        fig_signals.add_trace(go.Scatter(
            x=[s['confirm_date'] for s in short_signals],
            y=[s['confirm_price'] for s in short_signals],
            mode='markers', name=f'🔴 숏 ({len(short_signals)}회)',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
        
        fig_signals.update_layout(
            height=500, title="전체 기간 롱/숏 시그널"
        )
        
        st.plotly_chart(fig_signals, use_container_width=True)
    
    # ===== 탭 3: 전략 성과 =====
    with tab3:
        st.header("💹 전략 성과")
        
        if trades:
            long_trades = [t for t in trades if t['type'] == 'long']
            short_trades = [t for t in trades if t['type'] == 'short']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 전체 성과")
                total_trades = len(trades)
                wins = len([t for t in trades if t['return'] > 0])
                total_return = sum(t['return'] for t in trades)
                avg_return = total_return / total_trades if total_trades > 0 else 0
                
                st.metric("총 거래", f"{total_trades}회")
                st.metric("승률", f"{wins/total_trades*100:.1f}%")
                st.metric("평균 수익률", f"{avg_return:+.2f}%")
                st.metric("누적 수익률", f"{total_return:+.1f}%")
            
            with col2:
                st.subheader("🟢 롱 성과")
                if long_trades:
                    long_wins = len([t for t in long_trades if t['return'] > 0])
                    long_total = sum(t['return'] for t in long_trades)
                    long_avg = long_total / len(long_trades)
                    
                    st.metric("롱 거래", f"{len(long_trades)}회")
                    st.metric("롱 승률", f"{long_wins/len(long_trades)*100:.1f}%")
                    st.metric("롱 평균", f"{long_avg:+.2f}%")
                    st.metric("롱 누적", f"{long_total:+.1f}%")
                else:
                    st.info("롱 거래 없음")
            
            with col3:
                st.subheader("🔴 숏 성과")
                if short_trades:
                    short_wins = len([t for t in short_trades if t['return'] > 0])
                    short_total = sum(t['return'] for t in short_trades)
                    short_avg = short_total / len(short_trades)
                    
                    st.metric("숏 거래", f"{len(short_trades)}회")
                    st.metric("숏 승률", f"{short_wins/len(short_trades)*100:.1f}%")
                    st.metric("숏 평균", f"{short_avg:+.2f}%")
                    st.metric("숏 누적", f"{short_total:+.1f}%")
                else:
                    st.info("숏 거래 없음")
            
            st.divider()
            
            # 청산 사유별 분석
            st.subheader("📋 청산 사유별 분석")
            
            reason_stats = {}
            for t in trades:
                key = f"{t['type']}-{t['exit_reason']}"
                if key not in reason_stats:
                    reason_stats[key] = {'count': 0, 'returns': []}
                reason_stats[key]['count'] += 1
                reason_stats[key]['returns'].append(t['return'])
            
            reason_df = pd.DataFrame([{
                '포지션': '롱' if 'long' in key else '숏',
                '청산사유': key.split('-')[1],
                '횟수': stats['count'],
                '평균수익': f"{sum(stats['returns'])/len(stats['returns']):+.1f}%",
                '총수익': f"{sum(stats['returns']):+.1f}%"
            } for key, stats in reason_stats.items()])
            
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다")
    
    # ===== 탭 4: 거래 내역 =====
    with tab4:
        st.header("📉 거래 내역")
        
        if trades:
            sorted_trades = sorted(trades, key=lambda x: x['exit_date'], reverse=True)
            
            trade_df = pd.DataFrame([{
                '타입': '🟢 롱' if t['type'] == 'long' else '🔴 숏',
                '진입일': t['entry_dates'][0].strftime('%Y-%m-%d'),
                '진입횟수': f"{t['num_entries']}회",
                '평균가': f"${t['avg_price']:,.2f}",
                '청산일': t['exit_date'].strftime('%Y-%m-%d'),
                '청산가': f"${t['exit_price']:,.2f}",
                '수익률': f"{t['return']:+.1f}%",
                '청산사유': t['exit_reason']
            } for t in sorted_trades])
            
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다")
    
    # ===== 탭 5: 데이터 확인 =====
    with tab5:
        st.header("🔍 데이터 확인")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            days_diff = (today - last_date).days
            
            if days_diff <= 1:
                st.success(f"✅ 최신 데이터\n{last_date.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning(f"⚠️ {days_diff}일 전 데이터")
        
        with col2:
            missing = df['Close'].isna().sum()
            if missing == 0:
                st.success("✅ 결측치 없음")
            else:
                st.error(f"❌ 결측치 {missing}개")
        
        with col3:
            st.metric("총 데이터", f"{len(df)}봉")
        
        st.divider()
        
        st.subheader("📊 최근 데이터 (마지막 20봉)")
        
        recent_df = df.tail(20).copy()
        recent_df = recent_df.sort_index(ascending=False)
        
        display_df = pd.DataFrame({
            '시간': recent_df.index.strftime('%Y-%m-%d %H:%M'),
            '종가': recent_df['Close'].apply(lambda x: f"${x:,.2f}"),
            'RSI': recent_df['rsi'].apply(lambda x: f"{x:.1f}"),
            'MA100': recent_df['MA100'].apply(lambda x: f"${x:,.0f}" if not pd.isna(x) else "N/A"),
            'GC': recent_df['golden_cross'].apply(lambda x: "🟢" if x else "🔴")
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        if st.button("🔄 데이터 새로고침", type="primary"):
            st.cache_data.clear()
            st.success("✅ 캐시 삭제 완료!")
            st.rerun()


if __name__ == "__main__":
    main()

