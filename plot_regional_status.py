import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------------------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------------------------------
base_path = "/Users/limtae-kyu/culture-policy-card-movie/data"
card_df = pd.read_csv(f"{base_path}/panel_sido_month.csv")

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 2. 데이터 전처리
# --------------------------------------------------------------------------
# (1) 지역별 그룹화
# content_sales_2022: 연간 콘텐츠 산업 매출액 (단위가 큼)
# FNB, SHOP, CULTURE: 월별 카드 매출액
sido_status = card_df.groupby('SIDO_SHORT').agg({
    'content_sales_2022': 'mean',  # 연간 매출액 (모든 월에 같은 값이므로 평균=그 값)
    'FNB': 'mean',
    'SHOP': 'mean',
    'CULTURE': 'mean'
}).reset_index()

# (2) 카드 총 매출액 계산 (3개 업종 합계)
sido_status['Card_Total'] = sido_status['FNB'] + sido_status['SHOP'] + sido_status['CULTURE']

# (3) 정렬: 콘텐츠 산업 매출액 기준 내림차순 (큰 순서대로)
sido_status = sido_status.sort_values('content_sales_2022', ascending=False)

# --------------------------------------------------------------------------
# 3. 시각화 (Combo Chart: Bar + Line)
# --------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 7))

x = sido_status['SIDO_SHORT']

# [왼쪽 Y축] 콘텐츠 산업 매출액 (막대 그래프)
ax1.bar(x, sido_status['content_sales_2022'], color='#1F77B4', alpha=0.6, label='Content Industry Sales (2022)')
ax1.set_xlabel('Region (SIDO)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Content Industry Sales (KRW)', fontsize=12, fontweight='bold', color='#1F77B4')
ax1.tick_params(axis='y', labelcolor='#1F77B4')
# 단위를 '조' 단위로 변환하여 표시 (가독성)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000000000000:,.1f}조'))

# [오른쪽 Y축] 카드 총 매출액 (꺾은선 그래프)
ax2 = ax1.twinx()
ax2.plot(x, sido_status['Card_Total'], color='#D62728', linewidth=3, marker='o', markersize=8, label='Avg. Monthly Card Sales (Total)')
ax2.set_ylabel('Avg. Card Sales (KRW)', fontsize=12, fontweight='bold', color='#D62728')
ax2.tick_params(axis='y', labelcolor='#D62728')
# 단위를 '천억' 단위로 변환
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/100000000000:,.0f}천억'))

# 타이틀 및 범례
plt.title('Regional Comparison: Content Industry Scale vs Commercial Vitality', fontsize=16, pad=20)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88), bbox_transform=ax1.transAxes)

plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()