import os
import pandas as pd
import numpy as np

# =========================================
# 0. 파일 경로 설정 (레포 루트 기준)
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 여기 = 레포 루트
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일명들 (네가 말한 걸로 반영)
card_file    = os.path.join(DATA_DIR, "card_2024_2025_only.csv")
theater_file = os.path.join(DATA_DIR, "KC_497_DMSTC_MCST_THEART_2025.csv")
movie_file   = os.path.join(DATA_DIR, "BOX_OFFIC_MOVIE_2024_2025_ALL.csv")
output_panel = os.path.join(DATA_DIR, "panel_sido_month.csv")

# =========================================
# 1. 카드데이터 전처리 + 시도×월×업종 집계
# =========================================

chunksize = 500_000
card_agg_list = []

# 👉 GB2 → 분석용 상위 업종 그룹 매핑
def map_cat_gb2(gb2):
    if pd.isna(gb2):
        return "OTHER"

    if "외식" in gb2 or "음식" in gb2 or "식당" in gb2 or "카페" in gb2:
        return "FNB"          # 식음료/외식
    if "종합쇼핑" in gb2 or "패션쇼핑" in gb2 or "의류" in gb2:
        return "SHOP"         # 쇼핑 (특히 종합+패션)
    if "공연관람" in gb2 or "문화" in gb2:
        return "CULTURE"      # 넓은 의미의 공연/문화
    return "OTHER"


for chunk in pd.read_csv(
    card_file,
    chunksize=chunksize,
    encoding="utf-8-sig",
    engine="python"
):
    # TA_YM 문자열로 맞추기
    if 'TA_YM' not in chunk.columns:
        raise ValueError("카드 데이터에 'TA_YM' 컬럼이 없습니다. 컬럼명을 확인하세요.")
    chunk['TA_YM'] = chunk['TA_YM'].astype(str)

    # 분석 기간: 2024-01 ~ 2025-12만 사용
    chunk = chunk[(chunk['TA_YM'] >= '202401') & (chunk['TA_YM'] <= '202512')]
    if chunk.empty:
        continue

    # 시도 컬럼: '가맹점광역시도' 사용
    if '가맹점광역시도' not in chunk.columns:
        raise ValueError("카드 데이터에 '가맹점광역시도' 컬럼이 없습니다. 실제 컬럼명을 다시 확인하세요.")
    chunk['SIDO_SHORT'] = chunk['가맹점광역시도'].astype(str)

    # GB2 → 분석용 업종 그룹
    if 'GB2' not in chunk.columns:
        raise ValueError("카드 데이터에 'GB2' 컬럼이 없습니다.")
    chunk['CAT_GRP'] = chunk['GB2'].apply(map_cat_gb2)

    # 기본 집계 (시도×월×카테고리)
    grp = (chunk
           .groupby(['SIDO_SHORT', 'TA_YM', 'CAT_GRP'], as_index=False)
           .agg(
               VLM=('VLM', 'sum'),
               USEC=('USEC', 'sum')
           ))
    card_agg_list.append(grp)

# 조각들 합치기
card_agg = pd.concat(card_agg_list, ignore_index=True)

# 다시 한 번 전체 집계 (혹시 중복 있을 수 있으니)
card_agg = (card_agg
            .groupby(['SIDO_SHORT', 'TA_YM', 'CAT_GRP'], as_index=False)
            .agg(
                VLM=('VLM', 'sum'),
                USEC=('USEC', 'sum')
            ))

# 피벗: 시도×월 기준으로 FNB/CVS/TRANS 등을 컬럼으로
card_pivot = card_agg.pivot_table(
    index=['SIDO_SHORT', 'TA_YM'],
    columns='CAT_GRP',
    values='VLM',
    aggfunc='sum',
    fill_value=0
).reset_index()

card_pivot.columns.name = None  # 다중 인덱스 해제

# =========================================
# 2. 극장 데이터 → 시도별 극장 수 / Treat 더미
# =========================================

# 극장 CSV는 cp949라 cp949로 읽기
theater = pd.read_csv(theater_file, encoding="cp949")

# 극장 데이터의 시도 컬럼: 'sido_nm' (예: 서울특별시, 경상북도 등)
if 'sido_nm' not in theater.columns:
    raise ValueError("극장 데이터에 'sido_nm' 컬럼이 없습니다. 실제 컬럼명을 확인하세요.")

def normalize_sido_long_to_short(x: str) -> str:
    """
    '서울특별시' -> '서울', '경상북도' -> '경북' 형태로 줄이기
    (카드데이터 SIDO_SHORT='서울','경북' 등과 맞추기 위함)
    """
    if pd.isna(x):
        return x
    x = str(x)
    if x.endswith('특별시'):
        return x[:-3]
    if x.endswith('광역시'):
        return x[:-3]
    if x.endswith('특별자치시'):
        return x[:-5]
    if x.endswith('특별자치도'):
        return x[:-5]
    if x.endswith('도'):
        base = x[:-1]
        if len(base) >= 3 and base.endswith(('북', '남')):
            return base[-2:]   # '경상북' -> '경북'
        return base
    return x

theater['SIDO_SHORT'] = theater['sido_nm'].apply(normalize_sido_long_to_short)

gu_theater = (theater
              .groupby('SIDO_SHORT', as_index=False)
              .agg(THEATER_CNT=('id_poi', 'nunique')))

# Treat 구분: 극장 수 중앙값 이상이면 1, 아니면 0
med_cnt = gu_theater['THEATER_CNT'].median()
gu_theater['TREAT_SIDO'] = (gu_theater['THEATER_CNT'] >= med_cnt).astype(int)

# =========================================
# 3. 영화/박스오피스 → 블록버스터 개봉월 플래그
# =========================================

# 영화 통합 파일은 네가 utf-8-sig로 만들었을 가능성이 크니 우선 이걸 사용
movie = pd.read_csv(movie_file, encoding="utf-8-sig")

# 필수 컬럼 체크
for col in ['OPN_DE', 'TOT_SCRN_CO']:
    if col not in movie.columns:
        raise ValueError(f"영화 데이터에 '{col}' 컬럼이 없습니다. 실제 컬럼명을 확인하세요.")

# 개봉월 (YYYYMM) 추출
movie['OPN_DE'] = movie['OPN_DE'].astype(str)
movie['OPN_YM'] = movie['OPN_DE'].str.slice(0, 6)

# TOT_SCRN_CO 숫자형으로 변환 (혹시 문자열이면)
movie['TOT_SCRN_CO'] = pd.to_numeric(movie['TOT_SCRN_CO'], errors='coerce')

# 블록버스터 기준 정하기 (예: 스크린 수 상위 20%)
thr = movie['TOT_SCRN_CO'].quantile(0.95)
movie['IS_BLOCKBUSTER'] = (movie['TOT_SCRN_CO'] >= thr).astype(int)

# 월별 블록버스터 개수/존재 여부 집계
bb_month = (movie
            .groupby('OPN_YM', as_index=False)
            .agg(
                BB_MOVIE_CNT=('IS_BLOCKBUSTER', 'sum')
            ))
bb_month['BLOCKBUSTER_MONTH'] = (bb_month['BB_MOVIE_CNT'] > 0).astype(int)

# =========================================
# 4. 시도×월 패널 조인 (카드 + 극장 + 블록버스터)
# =========================================

panel = card_pivot.copy()  # SIDO_SHORT, TA_YM, FNB/CVS/...

# 극장 정보 조인 (SIDO_SHORT 기준)
panel = panel.merge(gu_theater, on='SIDO_SHORT', how='left')

# 블록버스터 월 조인 (TA_YM ↔ OPN_YM)
panel = panel.merge(
    bb_month[['OPN_YM', 'BB_MOVIE_CNT', 'BLOCKBUSTER_MONTH']],
    left_on='TA_YM',
    right_on='OPN_YM',
    how='left'
)

# 블록버스터 정보 없는 월(개봉 영화 없을 수도 있음)은 0으로
panel['BB_MOVIE_CNT'] = panel['BB_MOVIE_CNT'].fillna(0).astype(int)
panel['BLOCKBUSTER_MONTH'] = panel['BLOCKBUSTER_MONTH'].fillna(0).astype(int)

# DID 상호작용 변수
panel['DID'] = panel['TREAT_SIDO'] * panel['BLOCKBUSTER_MONTH']

# 로그 변환 예시 (FNB / CVS / TRANS 매출 기준)
if 'FNB' in panel.columns:
    panel['logVLM_FNB'] = np.log(panel['FNB'] + 1)
if 'CVS' in panel.columns:
    panel['logVLM_CVS'] = np.log(panel['CVS'] + 1)
if 'TRANS' in panel.columns:
    panel['logVLM_TRANS'] = np.log(panel['TRANS'] + 1)

# =========================================
# 4-1. 블록버스터/극장 Treat 분포 체크 (디버그/요약) 및 결측치 제거
# =========================================

print("\nNULL 값 제거")
panel = panel.dropna(subset=['TREAT_SIDO', 'BLOCKBUSTER_MONTH', 'logVLM_FNB'])

print("\n[BLOCKBUSTER_MONTH × TREAT_SIDO 교차표]")
print(pd.crosstab(panel['BLOCKBUSTER_MONTH'], panel['TREAT_SIDO'], dropna=False))

print("\n[월별 블록버스터 비율]")
print(panel.groupby('TA_YM')['BLOCKBUSTER_MONTH'].mean())

# =========================================
# 5. 결과 저장
# =========================================

panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
print("패널 데이터 저장 완료:", output_panel)

print("\n패널 샘플 10행:")
print(panel.head(10))
