"""
BTC 5년 데이터: 1달 보유 시 최대 낙폭 → 안전 레버리지 계산
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def main():
    print("=" * 60)
    print("₿ BTC 1달 보유 시 최대 낙폭 분석 (5년)")
    print("   → 청산 안 당할 최대 레버리지 계산")
    print("=" * 60)
    
    # 5년 일봉 데이터
    end = datetime.now()
    start = end - timedelta(days=365*5)
    
    print(f"\n📅 데이터 로딩 중...")
    df = yf.download('BTC-USD', start=start, end=end, interval='1d', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"📅 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"📊 데이터: {len(df)}일")
    
    # 각 시점에서 1달(30일) 보유 시 최대 낙폭 계산
    holding_days = 30
    max_drawdowns = []
    worst_cases = []
    
    print(f"\n🔍 각 시점에서 {holding_days}일 보유 시 최대 낙폭 계산 중...")
    
    for i in range(len(df) - holding_days):
        buy_price = df['Close'].iloc[i]
        buy_date = df.index[i]
        
        # 보유 기간 동안 최저가
        holding_period = df['Low'].iloc[i:i+holding_days]
        min_price = holding_period.min()
        min_date = holding_period.idxmin()
        
        # 최대 낙폭 (매수가 대비)
        mdd = (min_price / buy_price - 1) * 100
        
        max_drawdowns.append({
            'buy_date': buy_date,
            'buy_price': buy_price,
            'min_date': min_date,
            'min_price': min_price,
            'mdd': mdd
        })
    
    # 최악의 케이스들 정렬
    sorted_mdd = sorted(max_drawdowns, key=lambda x: x['mdd'])
    
    print("\n" + "=" * 60)
    print("📉 최악의 낙폭 TOP 10 (1달 보유 기준)")
    print("=" * 60)
    
    for i, case in enumerate(sorted_mdd[:10]):
        print(f"  {i+1}. 매수: {case['buy_date'].date()} ${case['buy_price']:,.0f}")
        print(f"      최저: {case['min_date'].date()} ${case['min_price']:,.0f} → 낙폭: {case['mdd']:.1f}%")
        print()
    
    # 최대 낙폭
    worst = sorted_mdd[0]
    worst_mdd = abs(worst['mdd'])
    
    print("=" * 60)
    print("🔴 최대 낙폭 (5년간 최악의 경우)")
    print("=" * 60)
    print(f"  매수일: {worst['buy_date'].date()}")
    print(f"  매수가: ${worst['buy_price']:,.0f}")
    print(f"  최저일: {worst['min_date'].date()}")
    print(f"  최저가: ${worst['min_price']:,.0f}")
    print(f"  낙폭: {worst['mdd']:.1f}%")
    
    # 안전 레버리지 계산
    print("\n" + "=" * 60)
    print("💰 안전 레버리지 계산")
    print("=" * 60)
    
    safe_leverage = 100 / worst_mdd
    
    print(f"\n  최대 낙폭: {worst_mdd:.1f}%")
    print(f"  청산 안 당할 최대 레버리지: {safe_leverage:.2f}배")
    print(f"\n  ⚠️ 안전 마진 20% 적용 시: {safe_leverage * 0.8:.2f}배")
    print(f"  ⚠️ 안전 마진 50% 적용 시: {safe_leverage * 0.5:.2f}배")
    
    # 레버리지별 청산 확률
    print("\n" + "=" * 60)
    print("📊 레버리지별 '1달 내 청산' 발생 횟수 (5년간)")
    print("=" * 60)
    
    for leverage in [2, 3, 5, 7, 10, 15, 20]:
        liquidation_threshold = -100 / leverage
        liquidated = sum(1 for m in max_drawdowns if m['mdd'] <= liquidation_threshold)
        pct = liquidated / len(max_drawdowns) * 100
        print(f"  {leverage:>2}배: 청산 {liquidated:>4}회 / {len(max_drawdowns)}회 ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("💡 결론")
    print("=" * 60)
    print(f"""
  5년간 BTC를 아무 때나 사서 1달 보유했을 때:
  
  최악의 경우 {worst_mdd:.1f}% 하락
  
  → {int(safe_leverage)}배 이하 레버리지면 청산 확률 0%
  → 안전하게 {int(safe_leverage * 0.5)}배 이하 권장
""")


if __name__ == '__main__':
    main()

