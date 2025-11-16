import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# 맥 기본 한글 폰트 설정
mpl.rc('font', family='AppleGothic')

# 음수 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 1. 데이터 로드 및 설정
# --------------------------------------------------------------------------
# 사용자 지정 경로
base_path = "/Users/limtae-kyu/culture-policy-card-movie/data"

# 파일 읽기
movie_df = pd.read_csv(f"{base_path}/movie_monthly_cumulative.csv")
card_df = pd.read_csv(f"{base_path}/panel_sido_month.csv")

# 한글 폰트 설정 (Mac OS의 경우 'AppleGothic', 윈도우는 'Malgun Gothic' 등 사용)
# plt.rcParams['font.family'] = 'AppleGothic'
# plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 2. 데이터 전처리
# --------------------------------------------------------------------------

# (1) 영화 데이터: 월별 총 매출액 집계
# YM 컬럼을 기준으로 그룹화하여 매출액(SALES_PRICE) 합계 계산
movie_monthly = movie_df.groupby('YM')['SALES_PRICE'].sum().reset_index()
movie_monthly['YM'] = pd.to_datetime(movie_monthly['YM'].astype(str), format='%Y%m')
movie_monthly.rename(columns={'SALES_PRICE': 'Movie_Sales'}, inplace=True)

# (2) 카드 데이터: 월별 업종별 총 매출액 집계
# TA_YM 컬럼을 기준으로 그룹화하여 각 업종(FNB, SHOP, CULTURE) 합계 계산
card_monthly = card_df.groupby('TA_YM')[['FNB', 'SHOP', 'CULTURE']].sum().reset_index()
card_monthly['TA_YM'] = pd.to_datetime(card_monthly['TA_YM'].astype(str), format='%Y%m')

# (3) 데이터 병합
merged_df = pd.merge(card_monthly, movie_monthly, left_on='TA_YM', right_on='YM', how='inner')

# --------------------------------------------------------------------------
# 3. 시각화 (이중 축 그래프)
# --------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(14, 8))

# X축 설정
x = merged_df['TA_YM']

# [왼쪽 Y축] 카드 매출 (선 그래프)
# 선 두께와 마커를 통해 가독성을 높임
line1 = ax1.plot(x, merged_df['FNB'], label='FNB (외식)', color='#FF7F0E', linewidth=2, marker='o')
line2 = ax1.plot(x, merged_df['SHOP'], label='SHOP (쇼핑)', color='#2CA02C', linewidth=2, marker='s')
line3 = ax1.plot(x, merged_df['CULTURE'], label='CULTURE (문화)', color='#1F77B4', linewidth=2, marker='^')

ax1.set_xlabel('Year-Month', fontsize=12)
ax1.set_ylabel('Card Sales (KRW)', fontsize=12, fontweight='bold')
# Y축 단위를 '억' 단위 등으로 보기 좋게 포맷팅
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/100000000:,.0f}억'))

# [오른쪽 Y축] 영화 매출 (막대 그래프)
# 투명도(alpha)를 주어 선 그래프를 가리지 않도록 설정
ax2 = ax1.twinx()
bar = ax2.bar(x, merged_df['Movie_Sales'], label='Movie Box Office', color='gray', alpha=0.3, width=20)

ax2.set_ylabel('Movie Box Office Sales (KRW)', fontsize=12, fontweight='bold', color='gray')
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/100000000:,.0f}억'))
ax2.tick_params(axis='y', labelcolor='gray')

# [범례 및 타이틀 설정]
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
# 막대 그래프 범례 추가
lines += [bar]
labels += [bar.get_label()]

ax1.legend(lines, labels, loc='upper left', fontsize=10, frameon=True)
plt.title('Monthly Trend: Movie Box Office vs Card Sales', fontsize=16, pad=20)
plt.grid(True, axis='x', linestyle='--')

# 그래프 출력
plt.tight_layout()
plt.show()