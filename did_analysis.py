import os
import pandas as pd
import statsmodels.formula.api as smf

# =========================================
# 0. 패널 데이터 불러오기
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 현재 파일이 있는 폴더 (레포 루트)
DATA_DIR = os.path.join(BASE_DIR, "data")

panel_file = os.path.join(DATA_DIR, "panel_sido_month.csv")

df = pd.read_csv(panel_file, encoding="utf-8-sig")

print("컬럼들:", df.columns.tolist())
print("\n상위 5행:")
print(df.head())

# =========================================
# 1. 기간 필터 (선택: 2024~2025만)
# =========================================

df['TA_YM'] = df['TA_YM'].astype(str)

df = df[(df['TA_YM'] >= '202401') & (df['TA_YM'] <= '202512')].copy()

print("\n연월 범위:", df['TA_YM'].min(), " ~ ", df['TA_YM'].max())
print("관측치 수:", len(df))

# =========================================
# 2. 기본 분포 확인
# =========================================

print("\n극장 Treat 더미 (TREAT_SIDO):")
print(df['TREAT_SIDO'].value_counts(dropna=False))

print("\n블록버스터 월 여부 (BLOCKBUSTER_MONTH):")
print(df['BLOCKBUSTER_MONTH'].value_counts(dropna=False))

print("\nDID 상호작용 (TREAT_SIDO * BLOCKBUSTER_MONTH):")
print(df['DID'].value_counts(dropna=False))

print("\n시도별 극장 수 요약:")
print(df[['SIDO_SHORT', 'THEATER_CNT']].drop_duplicates().sort_values('THEATER_CNT'))

print("\n월별 평균 블록버스터 평균 FNB:")
print(
    df.groupby('BLOCKBUSTER_MONTH')['FNB'].mean()
)

print("\nTreat vs Control 평균 지출:")
print(
    df.groupby('TREAT_SIDO')['FNB'].mean()
)



