import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 맥 기본 한글 폰트 설정
mpl.rc('font', family='AppleGothic')

# 음수 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False



# --------------------------------------------------------------------------
# 1. 데이터 로드 (경로 수정 완료)
# --------------------------------------------------------------------------
# 사용자님이 알려주신 로컬 경로
base_path = "/Users/limtae-kyu/culture-policy-card-movie/data"

# 경로를 포함하여 파일 읽기
card_df = pd.read_csv(f"{base_path}/panel_sido_month.csv")

# 한글 폰트 설정 (Mac 사용자로 추정되므로 AppleGothic 추천)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 2. 데이터 전처리
# --------------------------------------------------------------------------
# 지역별(SIDO_SHORT) 평균 계산
# log_content_sales: 2022년 기준 콘텐츠 산업 매출액의 로그값 (구조 변수)
# logVLM_FNB 등: 월별 카드 매출 로그값의 평균
sido_avg = card_df.groupby('SIDO_SHORT')[['log_content_sales', 'logVLM_FNB', 'logVLM_SHOP', 'logVLM_CULTURE']].mean().reset_index()

# --------------------------------------------------------------------------
# 3. 시각화 (Subplots)
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 시각화 대상 설정
targets = [
    ('logVLM_FNB', 'FNB (Dining)', '#FF7F0E'),      # 주황
    ('logVLM_SHOP', 'SHOP (Shopping)', '#2CA02C'),  # 초록
    ('logVLM_CULTURE', 'CULTURE (Leisure)', '#1F77B4') # 파랑
]

for ax, (col, label, color) in zip(axes, targets):
    # (1) 산점도 및 회귀선 그리기
    sns.regplot(x='log_content_sales', y=col, data=sido_avg, ax=ax,
                color=color,
                scatter_kws={'s': 100, 'alpha': 0.7, 'edgecolor': 'white'},
                line_kws={'color': 'gray', 'linestyle': '--', 'alpha': 0.8})

    # (2) 각 점에 지역명 텍스트 추가 (가독성을 위해 약간 오프셋)
    for i in range(sido_avg.shape[0]):
        ax.text(sido_avg.log_content_sales[i]+0.05, sido_avg[col][i],
                sido_avg.SIDO_SHORT[i], fontsize=9, fontweight='bold', alpha=0.8)

    # (3) 축 및 타이틀 꾸미기
    ax.set_title(f'Content Industry vs {label}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Log(Content Industry Sales)', fontsize=12)
    ax.set_ylabel(f'Log({label} Sales)', fontsize=12)

    # 그리드 추가
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Correlation: Regional Content Ecosystem & Commercial Sales', fontsize=18, y=1.05)
plt.tight_layout()
plt.show()