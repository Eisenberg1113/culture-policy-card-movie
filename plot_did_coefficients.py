import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

# 맥 기본 한글 폰트 설정
mpl.rc('font', family='AppleGothic')

# 음수 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------------------------------
# 사용자 지정 경로
base_path = "/Users/limtae-kyu/culture-policy-card-movie/data"
df_v1 = pd.read_csv(f"{base_path}/did_results_summary.csv")

# --------------------------------------------------------------------------
# 2. 데이터 전처리
# --------------------------------------------------------------------------
# 분석 결과 중 'DID'(이진 변수)와 'DID_INT'(강도 변수) 항만 추출
target_terms = ['DID', 'DID_INT']
plot_df = df_v1[df_v1['term'].isin(target_terms)].copy()

# 모델명 매핑 (보고서 용어에 맞춰 정리) 및 색상 지정
# FNB=주황, SHOP=초록, CULTURE=파랑 (시계열 그래프와 깔끔하게 통일)
model_mapping = {
    'A_binary_FNB_content': ('FNB (Binary DID)', '#FF7F0E'),
    'B_intensity_FNB_content': ('FNB (Intensity DID)', '#FF7F0E'),
    'C_intensity_leadlag_FNB': ('FNB (Lead-Lag)', '#FF7F0E'),
    'D_intensity_trend_FNB': ('FNB (Trend)', '#FF7F0E'),
    'E_intensity_hetero_FNB': ('FNB (Hetero)', '#FF7F0E'),
    'intensity_logVLM_SHOP': ('SHOP (Intensity DID)', '#2CA02C'),
    'intensity_logVLM_CULTURE': ('CULTURE (Intensity DID)', '#1F77B4')
}

# 매핑 적용
plot_df['label'] = plot_df['model'].map(lambda x: model_mapping.get(x, (x, 'black'))[0])
plot_df['color'] = plot_df['model'].map(lambda x: model_mapping.get(x, (x, 'black'))[1])

# 그래프 순서 정렬 (CULTURE -> SHOP -> FNB 순으로 쌓이도록 역순 정렬)
plot_df = plot_df.iloc[::-1].reset_index(drop=True)

# --------------------------------------------------------------------------
# 3. 시각화 (Forest Plot)
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# 각 모델별 계수와 신뢰구간 그리기
for i, row in plot_df.iterrows():
    # 오차 막대 (Error Bar): 95% 신뢰구간 (1.96 * SE)
    ax.errorbar(x=row['coef'], y=i, xerr=1.96*row['se'],
                fmt='o',                # 마커 모양 (원)
                color=row['color'],     # 색상
                ecolor=row['color'],    # 에러바 색상
                capsize=5,              # 에러바 끝 장식 크기
                elinewidth=2,           # 에러바 두께
                markersize=8)           # 마커 크기

    # 계수 값 텍스트 표시 (그래프 위에 숫자 표시)
    ax.text(row['coef'], i + 0.3, f"{row['coef']:.4f}",
            ha='center', va='bottom', fontsize=9, color=row['color'], fontweight='bold')

# 기준선 (x=0) 추가: 이 선에 신뢰구간이 걸치면 '유의하지 않음'
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# 축 설정
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df['label'], fontsize=11, fontweight='bold')
ax.set_xlabel('Coefficient Estimate (with 95% CI)', fontsize=12)
ax.set_title('DID Analysis Results: Impact of Blockbusters on Card Sales', fontsize=15, pad=15)

# 그리드 및 레이아웃
ax.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()

# 출력
plt.show()