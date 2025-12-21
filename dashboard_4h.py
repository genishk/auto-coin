"""
Auto-Coin 대시보드 (4시간봉)
streamlit run dashboard_4h.py

4시간봉 기준 분석 (하루 6번 체크)
- RSI 기반 매수/매도 시그널
- 물타기 전략 시뮬레이션
- 시그널 기준 슬라이더로 최적값 탐색
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
    page_title="Auto-Coin 4H",
    page_icon="⏰",
    layout="wide"
)


@st.cache_data(ttl=900)  # 15분 캐시 (4시간봉이라 더 자주 업데이트)
def load_data(ticker: str):
    """4시간봉 데이터 로드 및 지표 계산"""
    config = load_config()
    
    # 4시간봉용 별도 캐시
    cache = DataCache(
        cache_dir=str(project_root / "data" / "cache_4h"),
        max_age_hours=1  # 1시간마다 갱신
    )
    
    cache_key = f"{ticker}_4h"
    df = cache.get(cache_key)
    if df is None:
        fetcher = CoinFetcher([ticker])
        # 4시간봉, 2년 데이터
        data = fetcher.fetch(period='2y', interval='4h')
        if ticker in data:
            df = data[ticker]
            df, _ = validate_data(df, ticker)
            cache.set(cache_key, df)
    
    if df is not None:
        ti = TechnicalIndicators(config.get('indicators', {}))
        df = ti.calculate_all(df)
    
    return df


def find_buy_signals(df: pd.DataFrame, rsi_oversold: float = 30, rsi_exit: float = 50):
    """
    매수 시그널 찾기 (RSI 탈출 방식)
    조건: RSI < rsi_oversold 후 → RSI >= rsi_exit 탈출 시 매수
    """
    buy_signals = []
    
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    last_signal_rsi = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
            last_signal_rsi = rsi
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                buy_signals.append({
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'signal_rsi': last_signal_rsi,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': rsi
                })
                in_oversold = False
                last_signal_date = None
    
    return buy_signals


def find_sell_signals(df: pd.DataFrame, rsi_overbought: float = 70, rsi_exit: float = 50):
    """
    매도 시그널 찾기 (RSI 탈출 방식)
    조건: RSI > rsi_overbought 후 → RSI <= rsi_exit 하락 시 매도
    """
    sell_signals = []
    
    in_overbought = False
    last_signal_date = None
    last_signal_price = None
    last_signal_rsi = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
            last_signal_rsi = rsi
        else:
            if in_overbought and rsi <= rsi_exit and last_signal_date is not None:
                sell_signals.append({
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'signal_rsi': last_signal_rsi,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'confirm_rsi': rsi
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df: pd.DataFrame, buy_signals: list, sell_signals: list, stop_loss: float = -25):
    """
    물타기 전략 시뮬레이션 (수익일 때만 익절)
    - 매수 시그널 시 추가 매수 (물타기)
    - 매도 조건: 
      1) RSI 매도 시그널 + 수익인 경우 → 익절
      2) RSI 매도 시그널 + 손해인 경우 → 매도 안 함 (계속 보유)
      3) 손절 라인 도달 → 무조건 손절
    - confirm_date/confirm_price 기준 (실제 매수/매도 시점)
    """
    # confirm_date 기준으로 매수/매도 시점 결정 (실제 거래 시점)
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            total_cost = sum(p['price'] for p in positions)
            avg_price = total_cost / len(positions)
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = current_price
            
            # 1) 손절은 무조건 (최우선)
            if current_return <= stop_loss:
                exit_reason = "손절"
            # 2) RSI 매도 시그널 + 수익인 경우만 익절
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:  # 수익일 때만 매도!
                    exit_reason = "익절"
                    exit_price = sell_price
                # 손해면 매도하지 않음 (계속 보유)
            
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
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, positions


def main():
    st.title("⏰ Auto-Coin 4시간봉 분석")
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 사이드바
    st.sidebar.header("⚙️ 설정")
    
    config = load_config()
    tickers = config.get('tickers', ['BTC-USD'])
    ticker = st.sidebar.selectbox("코인", tickers, index=0)
    
    lookback_days = st.sidebar.slider("차트 기간 (일)", 7, 730, 180)  # 4시간봉: 최대 2년
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 RSI 기준 설정")
    
    # 매수 기준
    rsi_oversold = st.sidebar.slider("과매도 기준 (매수 시그널)", 10, 50, 35)
    rsi_buy_exit = st.sidebar.slider("매수 탈출 기준", 15, 100, 40)
    
    st.sidebar.markdown("---")
    
    # 매도 기준
    rsi_overbought = st.sidebar.slider("과매수 기준 (매도 시그널)", 50, 95, 80)
    rsi_sell_exit = st.sidebar.slider("매도 탈출 기준", 10, 70, 55)
    
    st.sidebar.markdown("---")
    stop_loss = st.sidebar.slider("손절 기준 (%)", -40, -10, -25)
    
    # 데이터 로드
    with st.spinner(f"{ticker} 데이터 로딩 중..."):
        df = load_data(ticker)
    
    if df is None:
        st.error(f"❌ {ticker} 데이터를 불러올 수 없습니다.")
        return
    
    st.sidebar.success(f"✅ {len(df)}일 데이터 로드")
    st.sidebar.info(f"📅 {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 시그널 계산
    buy_signals = find_buy_signals(df, rsi_oversold, rsi_buy_exit)
    sell_signals = find_sell_signals(df, rsi_overbought, rsi_sell_exit)
    trades, current_positions = simulate_trades(df, buy_signals, sell_signals, stop_loss)
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 현재 상태",
        "🔬 패턴 분석",
        "📈 RSI 분석",
        "🎯 매수/매도 시그널",
        "🔍 데이터 확인"
    ])
    
    # ===== 탭 1: 현재 상태 =====
    with tab1:
        st.header(f"📊 {ticker} 현재 상태")
        
        current = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = (current / prev - 1) * 100
        rsi_now = df['rsi'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("현재가", f"${current:,.2f}", f"{change:+.2f}%")
        with col2:
            rsi_status = "🔴 과매도" if rsi_now < rsi_oversold else ("🟢 과매수" if rsi_now > rsi_overbought else "⚪ 중립")
            st.metric("RSI", f"{rsi_now:.1f}", delta=rsi_status)
        with col3:
            if current_positions:
                avg_p = sum(p['price'] for p in current_positions) / len(current_positions)
                unrealized = (current / avg_p - 1) * 100
                st.metric("보유 상태", f"{len(current_positions)}회 물타기", delta=f"{unrealized:+.1f}%")
            else:
                st.metric("보유 상태", "대기 중")
        with col4:
            if trades:
                win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) * 100
                st.metric("전체 승률", f"{win_rate:.0f}%")
        
        st.divider()
        
        # 현재 포지션 상세
        if current_positions:
            st.subheader("💰 현재 보유 포지션")
            avg_price = sum(p['price'] for p in current_positions) / len(current_positions)
            unrealized = (current / avg_price - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 매수가", f"${avg_price:,.2f}")
            with col2:
                st.metric("물타기 횟수", f"{len(current_positions)}회")
            with col3:
                color = "🟢" if unrealized >= 0 else "🔴"
                st.metric("미실현 손익", f"{color} {unrealized:+.1f}%")
            
            st.markdown("**📋 매수 내역**")
            pos_df = pd.DataFrame([{
                '매수일': p['date'].strftime('%Y-%m-%d'),
                '매수가': f"${p['price']:,.2f}",
                '현재 손익': f"{(current/p['price']-1)*100:+.1f}%"
            } for p in current_positions])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            
            st.info(f"""
            **📤 매도 조건:**
            - RSI > {rsi_overbought} 발생 후 → RSI ≤ {rsi_sell_exit} 탈출 시 매도
            - 평단가 대비 {stop_loss}% 손절 (현재: {unrealized:+.1f}%)
            """)
        else:
            st.subheader("⏳ 대기 중")
            st.info("현재 보유 포지션이 없습니다. 매수 시그널 대기 중...")
        
        st.divider()
        
        # 최근 시그널
        st.subheader(f"🔔 시그널 내역 (최근 {lookback_days}일)")
        
        signal_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        filtered_buys = [bs for bs in buy_signals if bs['confirm_date'] >= signal_cutoff]
        filtered_sells = [ss for ss in sell_signals if ss['confirm_date'] >= signal_cutoff]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 매수 시그널** (실제 매수 시점)")
            if filtered_buys:
                buy_df = pd.DataFrame([{
                    '매수일': bs['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                    '매수가': f"${bs['confirm_price']:,.2f}",
                    '탈출RSI': f"{bs['confirm_rsi']:.1f}",
                    '시그널시작': bs['signal_date'].strftime('%m-%d'),
                } for bs in sorted(filtered_buys, key=lambda x: x['confirm_date'], reverse=True)])
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        with col2:
            st.markdown("**🔴 매도 시그널** (실제 매도 시점)")
            if filtered_sells:
                sell_df = pd.DataFrame([{
                    '매도일': ss['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                    '매도가': f"${ss['confirm_price']:,.2f}",
                    '탈출RSI': f"{ss['confirm_rsi']:.1f}",
                    '시그널시작': ss['signal_date'].strftime('%m-%d'),
                } for ss in sorted(filtered_sells, key=lambda x: x['confirm_date'], reverse=True)])
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        # RSI 상태 알림
        if rsi_now < rsi_oversold:
            st.warning(f"⚠️ RSI가 {rsi_oversold} 미만입니다 ({rsi_now:.1f}). 매수 시그널 구간!")
        elif rsi_now > rsi_overbought:
            st.warning(f"⚠️ RSI가 {rsi_overbought} 초과입니다 ({rsi_now:.1f}). 매도 시그널 구간!")
        
        st.divider()
        
        # 가격 차트 (실제 거래 결과 기반)
        st.subheader("📉 가격 차트 (실제 거래)")
        
        chart_df = df[df.index >= signal_cutoff]
        filtered_trades = [t for t in trades if t['exit_date'] >= signal_cutoff]
        
        fig_home = go.Figure()
        
        fig_home.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='가격'
        ))
        
        # 완료된 거래 표시
        for trade in filtered_trades:
            # 첫 매수 (초록색 삼각형)
            fig_home.add_trace(go.Scatter(
                x=[trade['entry_dates'][0]],
                y=[trade['entry_prices'][0]],
                mode='markers',
                marker=dict(color='limegreen', size=12, symbol='triangle-up',
                            line=dict(color='darkgreen', width=1)),
                showlegend=False,
                hovertemplate=f"🟢 매수: ${trade['entry_prices'][0]:,.2f}<br>{trade['entry_dates'][0].strftime('%Y-%m-%d %H:%M')}<extra></extra>"
            ))
            
            # 물타기 (연초록색 작은 원)
            if trade['num_buys'] > 1:
                for i in range(1, trade['num_buys']):
                    fig_home.add_trace(go.Scatter(
                        x=[trade['entry_dates'][i]],
                        y=[trade['entry_prices'][i]],
                        mode='markers',
                        marker=dict(color='lightgreen', size=8, symbol='circle',
                                    line=dict(color='green', width=1)),
                        showlegend=False,
                        hovertemplate=f"💧 물타기: ${trade['entry_prices'][i]:,.2f}<br>{trade['entry_dates'][i].strftime('%Y-%m-%d %H:%M')}<extra></extra>"
                    ))
            
            # 매도 (익절=파란색, 손절=빨간색)
            is_stoploss = '손절' in trade['exit_reason']
            sell_color = 'red' if is_stoploss else 'dodgerblue'
            sell_symbol = 'x' if is_stoploss else 'triangle-down'
            sell_label = '🔴 손절' if is_stoploss else '🔵 익절'
            
            fig_home.add_trace(go.Scatter(
                x=[trade['exit_date']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=sell_color, size=12, symbol=sell_symbol,
                            line=dict(color='darkblue' if not is_stoploss else 'darkred', width=1)),
                showlegend=False,
                hovertemplate=f"{sell_label}: ${trade['exit_price']:,.2f}<br>{trade['exit_date'].strftime('%Y-%m-%d %H:%M')}<br>수익률: {trade['return']:+.1f}%<extra></extra>"
            ))
        
        # 현재 보유 포지션 표시 (주황색)
        for pos in current_positions:
            if pos['date'] >= signal_cutoff:
                fig_home.add_trace(go.Scatter(
                    x=[pos['date']],
                    y=[pos['price']],
                    mode='markers',
                    marker=dict(color='orange', size=12, symbol='diamond',
                                line=dict(color='darkorange', width=1)),
                    showlegend=False,
                    hovertemplate=f"🟠 보유중: ${pos['price']:,.2f}<br>{pos['date'].strftime('%Y-%m-%d %H:%M')}<extra></extra>"
                ))
        
        # 범례 추가 (더미 트레이스)
        fig_home.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=10, symbol='triangle-up'), name='🟢 매수'))
        fig_home.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='lightgreen', size=8, symbol='circle'), name='💧 물타기'))
        fig_home.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='dodgerblue', size=10, symbol='triangle-down'), name='🔵 익절'))
        fig_home.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='red', size=10, symbol='x'), name='🔴 손절'))
        fig_home.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='orange', size=10, symbol='diamond'), name='🟠 보유중'))
        
        fig_home.update_layout(
            height=550,
            xaxis_rangeslider_visible=False,
            title=f"가격 차트 - 실제 거래 (최근 {lookback_days}일)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_home, use_container_width=True)
        
        st.divider()
        
        # 전략 성과
        filtered_trades = [t for t in trades if t['exit_date'] >= signal_cutoff]
        
        st.subheader(f"📈 전략 성과 (최근 {lookback_days}일)")
        
        if filtered_trades:
            total_trades = len(filtered_trades)
            wins = len([t for t in filtered_trades if t['return'] > 0])
            total_return = sum(t['return'] for t in filtered_trades)
            avg_return = total_return / total_trades
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 거래", f"{total_trades}회")
            with col2:
                st.metric("승률", f"{wins/total_trades*100:.0f}%")
            with col3:
                st.metric("평균 수익률", f"{avg_return:+.1f}%")
            with col4:
                st.metric("누적 수익률", f"{total_return:+.1f}%")
            
            st.markdown("**📋 거래 내역**")
            sorted_trades = sorted(filtered_trades, key=lambda x: x['exit_date'], reverse=True)
            trade_df = pd.DataFrame([{
                '기간': f"{t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}",
                '물타기': f"{t['num_buys']}회",
                '평단가': f"${t['avg_price']:,.2f}",
                '매도가': f"${t['exit_price']:,.2f}",
                '수익률': f"{t['return']:+.1f}%",
                '사유': t['exit_reason']
            } for t in sorted_trades])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"최근 {lookback_days}일간 완료된 거래 없음")
    
    # ===== 탭 2: 패턴 분석 (주식과 동일한 형식) =====
    with tab2:
        st.header("🔬 패턴 발생 분석")
        
        st.markdown("""
        **시그널 발생** vs **실제 매수 시그널** 구분
        - 연한 색: RSI 조건 충족 (시그널 발생)
        - 진한 색: RSI 탈출 확인 후 실제 매수/매도 시그널
        """)
        
        # ===== 매수 시그널 분석 =====
        st.subheader("🟢 매수 시그널 분석")
        st.caption(f"조건: RSI < {rsi_oversold} (과매도) → RSI ≥ X (탈출) 시 매수")
        
        # RSI 탈출 기준 슬라이더 (매수)
        buy_exit_slider = st.slider(
            "RSI 탈출 기준 (매수)", 
            15, 100, 40,
            help="과매도 구간 후 RSI가 이 값 이상이면 '매수 시그널'로 확정",
            key="buy_exit_slider"
        )
        
        # 모든 RSI 과매도 시점 (시그널 발생)
        all_oversold = []
        for idx in range(len(df)):
            rsi = df['rsi'].iloc[idx]
            if pd.notna(rsi) and rsi < rsi_oversold:
                all_oversold.append({
                    'date': df.index[idx],
                    'price': df['Close'].iloc[idx],
                    'rsi': rsi
                })
        
        # 실제 매수 시그널 (탈출 확인)
        actual_buy_signals = find_buy_signals(df, rsi_oversold, buy_exit_slider)
        buy_signal_dates = set(bs['signal_date'] for bs in actual_buy_signals)
        
        # 매수 시그널 차트
        fig_buy = go.Figure()
        
        # 가격 차트
        fig_buy.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1)
        ))
        
        # 일반 시그널 (연한 파란색) - 매수 시그널 제외
        normal_oversold = [s for s in all_oversold if s['date'] not in buy_signal_dates]
        fig_buy.add_trace(go.Scatter(
            x=[s['date'] for s in normal_oversold],
            y=[s['price'] for s in normal_oversold],
            mode='markers',
            name=f'시그널 발생 ({len(normal_oversold)}회)',
            marker=dict(color='lightblue', size=8, symbol='circle',
                        line=dict(color='blue', width=1)),
            hovertemplate='%{x}<br>가격: $%{y:,.2f}<br>RSI 시그널<extra></extra>'
        ))
        
        # 실제 매수 시그널 (진한 초록색) - confirm_date 기준 (실제 매수 시점!)
        fig_buy.add_trace(go.Scatter(
            x=[bs['confirm_date'] for bs in actual_buy_signals],
            y=[bs['confirm_price'] for bs in actual_buy_signals],
            mode='markers',
            name=f'★ 실제 매수 ({len(actual_buy_signals)}회)',
            marker=dict(color='limegreen', size=6, symbol='circle',
                        line=dict(color='darkgreen', width=1)),
            hovertemplate='%{x}<br>매수가: $%{y:,.2f}<br>★ 실제 매수 시점<extra></extra>'
        ))
        
        fig_buy.update_layout(
            title=f"매수 시그널: RSI < {rsi_oversold} → RSI ≥ {buy_exit_slider} 탈출",
            height=500,
            xaxis_title="날짜",
            yaxis_title="가격 ($)"
        )
        
        st.plotly_chart(fig_buy, use_container_width=True)
        
        # 매수 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 시그널", f"{len(all_oversold)}회")
        with col2:
            st.metric("★ 매수 시그널", f"{len(actual_buy_signals)}회", 
                      delta=f"RSI {buy_exit_slider}+ 탈출 확인")
        with col3:
            reduction = (1 - len(actual_buy_signals) / len(all_oversold)) * 100 if all_oversold else 0
            st.metric("필터링 비율", f"{reduction:.0f}% 감소")
        
        # 최근 매수 시그널 리스트
        if actual_buy_signals:
            st.markdown("**★ 최근 매수 시그널**")
            recent_buys = sorted(actual_buy_signals, key=lambda x: x['confirm_date'], reverse=True)[:10]
            buy_table = pd.DataFrame([{
                '★실제 매수일': bs['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                '★매수가': f"${bs['confirm_price']:,.2f}",
                'RSI': f"{bs['confirm_rsi']:.1f}",
                '(참고)과매도일': bs['signal_date'].strftime('%m-%d'),
            } for bs in recent_buys])
            st.dataframe(buy_table, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ===== 매도 시그널 분석 =====
        st.subheader("🔴 매도 시그널 분석")
        st.caption(f"조건: RSI > {rsi_overbought} (과매수) → RSI ≤ X (탈출) 시 매도")
        
        # RSI 탈출 기준 슬라이더 (매도)
        sell_exit_slider = st.slider(
            "RSI 탈출 기준 (매도)", 
            10, 70, 55,
            help="과매수 구간 후 RSI가 이 값 이하이면 '매도 시그널'로 확정",
            key="sell_exit_slider"
        )
        
        # 모든 RSI 과매수 시점 (시그널 발생)
        all_overbought = []
        for idx in range(len(df)):
            rsi = df['rsi'].iloc[idx]
            if pd.notna(rsi) and rsi > rsi_overbought:
                all_overbought.append({
                    'date': df.index[idx],
                    'price': df['Close'].iloc[idx],
                    'rsi': rsi
                })
        
        # 실제 매도 시그널 (탈출 확인)
        actual_sell_signals = find_sell_signals(df, rsi_overbought, sell_exit_slider)
        sell_signal_dates = set(ss['signal_date'] for ss in actual_sell_signals)
        
        # 매도 시그널 차트
        fig_sell = go.Figure()
        
        # 가격 차트
        fig_sell.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1)
        ))
        
        # 일반 시그널 (연한 빨간색) - 매도 시그널 제외
        normal_overbought = [s for s in all_overbought if s['date'] not in sell_signal_dates]
        fig_sell.add_trace(go.Scatter(
            x=[s['date'] for s in normal_overbought],
            y=[s['price'] for s in normal_overbought],
            mode='markers',
            name=f'시그널 발생 ({len(normal_overbought)}회)',
            marker=dict(color='lightsalmon', size=8, symbol='circle',
                        line=dict(color='red', width=1)),
            hovertemplate='%{x}<br>가격: $%{y:,.2f}<br>RSI 시그널<extra></extra>'
        ))
        
        # 실제 매도 시그널 (진한 빨간색) - confirm_date 기준 (실제 매도 시점!)
        fig_sell.add_trace(go.Scatter(
            x=[ss['confirm_date'] for ss in actual_sell_signals],
            y=[ss['confirm_price'] for ss in actual_sell_signals],
            mode='markers',
            name=f'★ 실제 매도 ({len(actual_sell_signals)}회)',
            marker=dict(color='red', size=6, symbol='circle',
                        line=dict(color='darkred', width=1)),
            hovertemplate='%{x}<br>매도가: $%{y:,.2f}<br>★ 실제 매도 시점<extra></extra>'
        ))
        
        fig_sell.update_layout(
            title=f"매도 시그널: RSI > {rsi_overbought} → RSI ≤ {sell_exit_slider} 탈출",
            height=500,
            xaxis_title="날짜",
            yaxis_title="가격 ($)"
        )
        
        st.plotly_chart(fig_sell, use_container_width=True)
        
        # 매도 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 시그널", f"{len(all_overbought)}회")
        with col2:
            st.metric("★ 매도 시그널", f"{len(actual_sell_signals)}회",
                      delta=f"RSI {sell_exit_slider} 이하 탈출")
        with col3:
            sell_reduction = (1 - len(actual_sell_signals) / len(all_overbought)) * 100 if all_overbought else 0
            st.metric("필터링 비율", f"{sell_reduction:.0f}% 감소")
        
        # 최근 매도 시그널 리스트
        if actual_sell_signals:
            st.markdown("**★ 최근 매도 시그널**")
            recent_sells = sorted(actual_sell_signals, key=lambda x: x['confirm_date'], reverse=True)[:10]
            sell_table = pd.DataFrame([{
                '★실제 매도일': ss['confirm_date'].strftime('%Y-%m-%d %H:%M'),
                '★매도가': f"${ss['confirm_price']:,.2f}",
                'RSI': f"{ss['confirm_rsi']:.1f}",
                '(참고)과매수일': ss['signal_date'].strftime('%m-%d'),
            } for ss in recent_sells])
            st.dataframe(sell_table, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ===== 통합 차트 =====
        st.subheader("🎯 매수/매도 시그널 통합")
        
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1.5)
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=[bs['confirm_date'] for bs in actual_buy_signals],
            y=[bs['confirm_price'] for bs in actual_buy_signals],
            mode='markers',
            name=f'🟢 실제 매수 ({len(actual_buy_signals)}회)',
            marker=dict(color='limegreen', size=8, symbol='circle',
                        line=dict(color='darkgreen', width=1))
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=[ss['confirm_date'] for ss in actual_sell_signals],
            y=[ss['confirm_price'] for ss in actual_sell_signals],
            mode='markers',
            name=f'🔴 실제 매도 ({len(actual_sell_signals)}회)',
            marker=dict(color='red', size=8, symbol='circle',
                        line=dict(color='darkred', width=1))
        ))
        
        fig_combined.update_layout(
            title=f"매수 (RSI {rsi_oversold}→{buy_exit_slider}) + 매도 (RSI {rsi_overbought}→{sell_exit_slider})",
            height=600,
            xaxis_title="날짜",
            yaxis_title="가격 ($)"
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # 통합 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 매수 시그널", f"{len(actual_buy_signals)}회")
        with col2:
            st.metric("🔴 매도 시그널", f"{len(actual_sell_signals)}회")
        with col3:
            if actual_buy_signals:
                years = (df.index[-1] - df.index[0]).days / 365
                st.metric("연간 매수", f"~{len(actual_buy_signals)/years:.1f}회")
        with col4:
            st.metric("손절 기준", f"{stop_loss}%")
    
    # ===== 탭 3: RSI 분석 =====
    with tab3:
        st.header("📈 RSI 기준 분석")
        
        st.markdown(f"""
        **현재 설정:**
        - 매수: RSI < **{rsi_oversold}** → RSI ≥ **{rsi_buy_exit}** 탈출 시
        - 매도: RSI > **{rsi_overbought}** → RSI ≤ **{rsi_sell_exit}** 탈출 시
        - 손절: **{stop_loss}%**
        
        *사이드바에서 기준값을 조절하면서 최적 값을 찾아보세요!*
        """)
        
        analysis_df = df.iloc[-lookback_days:] if lookback_days < len(df) else df
        
        # RSI 과매도/과매수 발생 횟수
        oversold_count = (analysis_df['rsi'] < rsi_oversold).sum()
        overbought_count = (analysis_df['rsi'] > rsi_overbought).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("과매도 일수", f"{oversold_count}일")
        with col2:
            st.metric("과매수 일수", f"{overbought_count}일")
        with col3:
            st.metric("매수 시그널", f"{len([b for b in buy_signals if b['signal_date'] >= analysis_df.index[0]])}회")
        with col4:
            st.metric("매도 시그널", f"{len([s for s in sell_signals if s['signal_date'] >= analysis_df.index[0]])}회")
        
        # RSI 차트
        fig_rsi = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(f'{ticker} 가격', 'RSI')
        )
        
        fig_rsi.add_trace(
            go.Scatter(x=analysis_df.index, y=analysis_df['Close'], name='가격',
                      line=dict(color='#1f77b4', width=1.5)),
            row=1, col=1
        )
        
        # RSI 과매도 시점
        oversold_dates = analysis_df[analysis_df['rsi'] < rsi_oversold].index
        if len(oversold_dates) > 0:
            fig_rsi.add_trace(
                go.Scatter(
                    x=oversold_dates,
                    y=analysis_df.loc[oversold_dates, 'Close'],
                    mode='markers',
                    name=f'RSI < {rsi_oversold}',
                    marker=dict(color='green', size=6, symbol='circle')
                ),
                row=1, col=1
            )
        
        # RSI 과매수 시점
        overbought_dates = analysis_df[analysis_df['rsi'] > rsi_overbought].index
        if len(overbought_dates) > 0:
            fig_rsi.add_trace(
                go.Scatter(
                    x=overbought_dates,
                    y=analysis_df.loc[overbought_dates, 'Close'],
                    mode='markers',
                    name=f'RSI > {rsi_overbought}',
                    marker=dict(color='red', size=6, symbol='circle')
                ),
                row=1, col=1
            )
        
        # RSI
        fig_rsi.add_trace(
            go.Scatter(x=analysis_df.index, y=analysis_df['rsi'], name='RSI',
                      line=dict(color='purple', width=1.5)),
            row=2, col=1
        )
        
        fig_rsi.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1,
                         annotation_text=f"과매수 ({rsi_overbought})")
        fig_rsi.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1,
                         annotation_text=f"과매도 ({rsi_oversold})")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig_rsi.update_layout(height=600, showlegend=True)
        fig_rsi.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_rsi.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # RSI 분포
        st.subheader("📊 RSI 분포")
        fig_hist = px.histogram(analysis_df, x='rsi', nbins=50, 
                                title='RSI 분포 히스토그램')
        fig_hist.add_vline(x=rsi_oversold, line_dash="dash", line_color="green",
                          annotation_text=f"과매도 ({rsi_oversold})")
        fig_hist.add_vline(x=rsi_overbought, line_dash="dash", line_color="red",
                          annotation_text=f"과매수 ({rsi_overbought})")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ===== 탭 3: 매수/매도 시그널 =====
    with tab4:
        st.header("🎯 매수/매도 시그널 분석")
        
        st.markdown(f"""
        **시그널 조건:**
        - 🟢 매수: RSI < {rsi_oversold} → RSI ≥ {rsi_buy_exit} 탈출
        - 🔴 매도: RSI > {rsi_overbought} → RSI ≤ {rsi_sell_exit} 탈출
        - ⛔ 손절: 평단가 대비 {stop_loss}%
        """)
        
        # 통합 시그널 차트
        fig_signals = go.Figure()
        
        fig_signals.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1.5)
        ))
        
        fig_signals.add_trace(go.Scatter(
            x=[bs['confirm_date'] for bs in buy_signals],
            y=[bs['confirm_price'] for bs in buy_signals],
            mode='markers',
            name=f'🟢 실제 매수 ({len(buy_signals)}회)',
            marker=dict(color='limegreen', size=10, symbol='triangle-up',
                        line=dict(color='darkgreen', width=2)),
            hovertemplate='%{x}<br>매수: $%{y:,.2f}<extra>🟢 매수</extra>'
        ))
        
        fig_signals.add_trace(go.Scatter(
            x=[ss['confirm_date'] for ss in sell_signals],
            y=[ss['confirm_price'] for ss in sell_signals],
            mode='markers',
            name=f'🔴 실제 매도 ({len(sell_signals)}회)',
            marker=dict(color='red', size=10, symbol='triangle-down',
                        line=dict(color='darkred', width=2)),
            hovertemplate='%{x}<br>매도: $%{y:,.2f}<extra>🔴 매도</extra>'
        ))
        
        fig_signals.update_layout(
            title="전체 기간 매수/매도 시그널",
            height=600,
            xaxis_title="날짜",
            yaxis_title="가격 ($)"
        )
        
        st.plotly_chart(fig_signals, use_container_width=True)
        
        # 통계
        st.subheader("📊 시그널 통계")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 총 매수 시그널", f"{len(buy_signals)}회")
        with col2:
            st.metric("🔴 총 매도 시그널", f"{len(sell_signals)}회")
        with col3:
            if buy_signals:
                years = (df.index[-1] - df.index[0]).days / 365
                st.metric("연간 매수 시그널", f"~{len(buy_signals)/years:.1f}회")
        
        st.divider()
        
        # 물타기 시뮬레이션 차트
        st.subheader("🎯 물타기 시뮬레이션")
        
        fig_strategy = go.Figure()
        
        fig_strategy.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1.5)
        ))
        
        for trade in trades:
            for i, (buy_date, buy_price) in enumerate(zip(trade['entry_dates'], trade['entry_prices'])):
                size = 14 if i == 0 else 10
                fig_strategy.add_trace(go.Scatter(
                    x=[buy_date],
                    y=[buy_price],
                    mode='markers',
                    marker=dict(color='limegreen', size=size, symbol='triangle-up',
                                line=dict(color='darkgreen', width=2)),
                    showlegend=False,
                    hovertemplate=f"{'매수' if i == 0 else '물타기'}: ${buy_price:,.2f}<br>{buy_date.strftime('%Y-%m-%d')}<extra></extra>"
                ))
            
            if trade['num_buys'] > 1:
                fig_strategy.add_trace(go.Scatter(
                    x=[trade['entry_dates'][0], trade['exit_date']],
                    y=[trade['avg_price'], trade['avg_price']],
                    mode='lines',
                    line=dict(color='orange', width=1, dash='dash'),
                    showlegend=False,
                    hovertemplate=f"평단: ${trade['avg_price']:,.2f}<extra></extra>"
                ))
            
            sell_color = 'red' if trade['return'] < 0 else 'blue'
            fig_strategy.add_trace(go.Scatter(
                x=[trade['exit_date']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=sell_color, size=14, symbol='triangle-down',
                            line=dict(color='darkred' if trade['return'] < 0 else 'darkblue', width=2)),
                showlegend=False,
                hovertemplate=f"매도: ${trade['exit_price']:,.2f}<br>{trade['exit_date'].strftime('%Y-%m-%d')}<br>{trade['exit_reason']}<br>수익률: {trade['return']:+.1f}%<extra></extra>"
            ))
        
        # 현재 포지션 표시
        if current_positions:
            avg_price = sum(p['price'] for p in current_positions) / len(current_positions)
            for i, p in enumerate(current_positions):
                size = 16 if i == 0 else 12
                fig_strategy.add_trace(go.Scatter(
                    x=[p['date']],
                    y=[p['price']],
                    mode='markers',
                    marker=dict(color='yellow', size=size, symbol='star',
                                line=dict(color='orange', width=2)),
                    showlegend=False,
                    hovertemplate=f"보유 중: ${p['price']:,.2f}<extra></extra>"
                ))
        
        # 범례
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=12, symbol='triangle-up'),
            name='🟢 매수/물타기'))
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='blue', size=12, symbol='triangle-down'),
            name='🔵 익절'))
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='🔴 손절'))
        
        fig_strategy.update_layout(
            title="물타기 시뮬레이션",
            height=650,
            xaxis_title="날짜",
            yaxis_title="가격 ($)"
        )
        
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # 거래 결과
        if trades:
            st.markdown("**📊 전체 기간 거래 결과**")
            
            total_trades = len(trades)
            wins = [t for t in trades if t['return'] > 0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 거래", f"{total_trades}회")
            with col2:
                win_rate = len(wins) / total_trades * 100
                st.metric("승률", f"{win_rate:.0f}%")
            with col3:
                avg_return = sum(t['return'] for t in trades) / total_trades
                st.metric("평균 수익률", f"{avg_return:+.1f}%")
            with col4:
                total_return = sum(t['return'] for t in trades)
                st.metric("총 수익률", f"{total_return:+.1f}%")
            
            st.markdown("**📋 거래 내역**")
            trade_df = pd.DataFrame([{
                '첫 매수일': t['entry_dates'][0].strftime('%Y-%m-%d'),
                '매수 횟수': f"{t['num_buys']}회",
                '평단가': f"${t['avg_price']:,.2f}",
                '매도일': t['exit_date'].strftime('%Y-%m-%d'),
                '매도가': f"${t['exit_price']:,.2f}",
                '수익률': f"{t['return']:+.1f}%",
                '매도 사유': t['exit_reason']
            } for t in sorted(trades, key=lambda x: x['entry_dates'][0], reverse=True)])
            
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
    
    # ===== 탭 4: 데이터 확인 =====
    with tab5:
        st.header("🔍 데이터 확인")
        
        cache_dir = project_root / "data" / "cache"
        metadata_file = cache_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if ticker in metadata:
                cache_info = metadata[ticker]
                cached_at = cache_info.get('cached_at', 'N/A')
                
                st.success(f"✅ 데이터 캐시 정상")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("캐시 저장 시간", cached_at[:19].replace('T', ' '))
                with col2:
                    st.metric("총 거래일", f"{cache_info.get('rows', 'N/A')}일")
                with col3:
                    st.metric("데이터 기간", f"{cache_info.get('start_date', '')} ~ {cache_info.get('end_date', '')}")
        else:
            st.warning("캐시 메타데이터 없음")
        
        st.divider()
        
        st.subheader(f"📊 최근 데이터 (마지막 30일)")
        
        recent_df = df.tail(30).copy()
        recent_df = recent_df.sort_index(ascending=False)
        
        display_df = pd.DataFrame({
            '날짜': recent_df.index.strftime('%Y-%m-%d'),
            '시가': recent_df['Open'].apply(lambda x: f"${x:,.2f}"),
            '고가': recent_df['High'].apply(lambda x: f"${x:,.2f}"),
            '저가': recent_df['Low'].apply(lambda x: f"${x:,.2f}"),
            '종가': recent_df['Close'].apply(lambda x: f"${x:,.2f}"),
            '거래량': recent_df['Volume'].apply(lambda x: f"{x/1e9:.2f}B"),
            'RSI': recent_df['rsi'].apply(lambda x: f"{x:.1f}"),
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("🔒 데이터 무결성")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            days_diff = (today - last_date).days
            
            if days_diff <= 1:
                st.success(f"✅ 최신 데이터\n마지막: {last_date.strftime('%Y-%m-%d')}")
            elif days_diff <= 3:
                st.warning(f"⚠️ {days_diff}일 전 데이터")
            else:
                st.error(f"❌ {days_diff}일 전 데이터\n업데이트 필요!")
        
        with col2:
            missing = df['Close'].isna().sum()
            if missing == 0:
                st.success(f"✅ 결측치 없음")
            else:
                st.error(f"❌ 결측치 {missing}개")
        
        with col3:
            total_rows = len(df)
            if total_rows >= 1000:
                st.success(f"✅ 충분한 데이터\n{total_rows}일")
            else:
                st.warning(f"⚠️ 데이터 부족?\n{total_rows}일")
        
        st.divider()
        
        st.subheader("🔄 데이터 새로고침")
        
        if st.button("🔄 지금 새로고침", type="primary"):
            cache = DataCache(str(cache_dir))
            cache.clear(ticker)
            st.cache_data.clear()
            st.success("✅ 캐시 삭제 완료!")
            st.rerun()


if __name__ == "__main__":
    main()
