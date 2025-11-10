import os
import pandas as pd
import numpy as np

# =========================================
# 0. 파일 경로 설정 (레포 기준 상대경로)
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")

card_file   = os.path.join(DATA_DIR, "card_2024_2025_only.csv")
theater_file = os.path.join(DATA_DIR, "KC_497_DMSTC_MCST_THEART_2025.csv")
movie_file   = os.path.join(DATA_DIR, "BOX_OFFIC_MOVIE_2024_2025_ALL.csv")  # 또는 통합 파일명
output_panel = os.path.join(DATA_DIR, "panel_sido_month.csv")

# =========================================
# 1. 카드데이터 전처리 + 시도×월×업종 집계
# =========================================

# ⚠ 여기는 네 카드데이터 컬럼명에 맞춰야 함
#   예시 가정:
#   - '가맹시' : 시도(서울, 부산, 경북 등)
#   - 'GB3'    : 상위 업종
#   - 'GB2'    : 하위 업종
#   - 'TA_YM'  : 연월 (예: 202301)
#   - 'VLM'    : 금액
#   - 'USEC'   : 건수

chunksize = 500_000

card_agg_list = []

# 👉 GB3 → 우리가 사용할 업종 그룹(카테고리)로 매핑
#    실제 GB3 값에 맞게 꼭 수정해서 쓰기!
def map_cat_gb3(gb3):
    """
    GB3 문자열을 받아서 분석용 카테고리로 매핑.
    - 예시는 대충 넣어둔 거라, 반드시 실제 값에 맞게 수정해야 함.
    """
    if pd.isna(gb3):
        return "OTHER"

    # 아래는 예시. 너 GB3 값 보고 수정해.
    if "음식" in gb3 or "외식" in gb3 or "식당" in gb3 or "카페" in gb3:
        return "FNB"       # Food & Beverage
    if "편의점" in gb3:
        return "CVS"       # 편의점
    if "교통" in gb3 or "주차" in gb3 or "택시" in gb3:
        return "TRANS"     # 교통
    if "문화" in gb3 or "공연" in gb3 or "영화" in gb3:
        return "CULTURE"   # 문화/공연
    # 스포츠활동/여행은 미리 필터에서 뺐다고 했으니 여기선 안 다룸
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

    # 혹시 2023 이전 데이터가 섞여 있으면 안전하게 한 번 더 필터
    chunk = chunk[(chunk['TA_YM'] >= '202301') & (chunk['TA_YM'] <= '202512')]

    if chunk.empty:
        continue

    # 시도 컬럼 이름 맞추기 (가맹점광역시도 또는 가맹시 사용)
    if '가맹점광역시도' in chunk.columns:
        chunk['SIDO_SHORT'] = chunk['가맹점광역시도'].astype(str)
    elif '가맹시' in chunk.columns:
        chunk['SIDO_SHORT'] = chunk['가맹시'].astype(str)
    else:
        raise ValueError("카드 데이터에 시도 컬럼이 없습니다. '가맹점광역시도' 또는 '가맹시' 컬럼명을 확인하세요.")

    # GB3 → 분석용 카테고리
    if 'GB3' not in chunk.columns:
        raise ValueError("카드 데이터에 'GB3' 컬럼이 없습니다.")
    chunk['CAT_GRP'] = chunk['GB3'].apply(map_cat_gb3)

    # 기본 집계 (시도×월×카테고리)
    grp = (chunk
           .groupby(['가맹시', 'TA_YM', 'CAT_GRP'], as_index=False)
           .agg(
               VLM=('VLM', 'sum'),
               USEC=('USEC', 'sum')
           ))
    card_agg_list.append(grp)

# 조각들 합치기
card_agg = pd.concat(card_agg_list, ignore_index=True)

# 다시 한 번 전체 집계 (혹시 중복 있을 수 있으니)
card_agg = (card_agg
            .groupby(['가맹시', 'TA_YM', 'CAT_GRP'], as_index=False)
            .agg(
                VLM=('VLM', 'sum'),
                USEC=('USEC', 'sum')
            ))

# 피벗: 시도×월 기준으로 FNB/CVS/TRANS 등을 컬럼으로
card_pivot = card_agg.pivot_table(
    index=['가맹시', 'TA_YM'],
    columns='CAT_GRP',
    values='VLM',
    aggfunc='sum',
    fill_value=0
).reset_index()

# 컬럼명 정리: 다중 인덱스 해제
card_pivot.columns.name = None

# 평균 객단가까지 보고 싶으면 USEC도 같은 방식으로 피벗 가능
# 여기서는 매출(VLM) 위주로만 예시 진행

# =========================================
# 2. 극장 데이터 → 시도별 극장 수 / Treat 더미
# =========================================

theater = pd.read_csv(theater_file, encoding="utf-8-sig")

# 극장 데이터의 시도 컬럼: sido_nm (예: 서울특별시, 경상북도 등)
if 'sido_nm' not in theater.columns:
    raise ValueError("극장 데이터에 'sido_nm' 컬럼이 없습니다.")

def normalize_sido_long_to_short(x: str) -> str:
    """
    '서울특별시' -> '서울', '경상북도' -> '경북' 형태로 줄이기
    (카드데이터 '가맹시'가 이런 형식이라고 가정)
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
        # 경상북도 -> 경상북 -> 경북 (2글자 줄이기)
        base = x[:-1]
        # 경상북, 경상남 등은 뒤 1글자만 남기면 '북','남'이라서 애매 → 2글자만 남기기
        if len(base) >= 3 and base.endswith(('북', '남')):
            return base[-2:]   # '경상북' -> '경북'
        return base
    return x

theater['SIDO_SHORT'] = theater['sido_nm'].apply(normalize_sido_long_to_short)

gu_theater = (theater
              .groupby('SIDO_SHORT', as_index=False)
              .agg(THEATER_CNT=('id_poi', 'nunique')))

# Treat 구분: 극장 수 중앙값 이상이면 1, 아니면 0 (필요에 따라 규칙 바꿔도 됨)
med_cnt = gu_theater['THEATER_CNT'].median()
gu_theater['TREAT_SIDO'] = (gu_theater['THEATER_CNT'] >= med_cnt).astype(int)

# =========================================
# 3. 영화/박스오피스 → 블록버스터 개봉월 플래그
# =========================================

movie = pd.read_csv(movie_file, encoding="utf-8-sig")

# 필수 컬럼 체크
for col in ['OPN_DE', 'TOT_SCRN_CO']:
    if col not in movie.columns:
        raise ValueError(f"영화 데이터에 '{col}' 컬럼이 없습니다.")

# 개봉월 (YYYYMM) 추출
movie['OPN_DE'] = movie['OPN_DE'].astype(str)
movie['OPN_YM'] = movie['OPN_DE'].str.slice(0, 6)

# 블록버스터 기준 정하기 (예: 스크린 수 상위 20%)
thr = movie['TOT_SCRN_CO'].quantile(0.8)

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

# 카드 피벗 테이블에 극장 Treat, 블록버스터 플래그 조인
panel = card_pivot.copy()

# 시도 이름을 SIDO_SHORT로 해석 (이미 '가맹시'가 '서울','경북' 형식이라고 가정)
panel = panel.rename(columns={'가맹시': 'SIDO_SHORT'})

# 극장 정보 조인
panel = panel.merge(gu_theater, on='SIDO_SHORT', how='left')

# 블록버스터 월 조인 (TA_YM ↔ OPN_YM)
panel = panel.merge(bb_month[['OPN_YM', 'BB_MOVIE_CNT', 'BLOCKBUSTER_MONTH']],
                    left_on='TA_YM', right_on='OPN_YM', how='left')

# 블록버스터 정보 없는 월(개봉 영화 없을 수도 있음)은 0으로
panel['BB_MOVIE_CNT'] = panel['BB_MOVIE_CNT'].fillna(0).astype(int)
panel['BLOCKBUSTER_MONTH'] = panel['BLOCKBUSTER_MONTH'].fillna(0).astype(int)

# DID 상호작용 변수
panel['DID'] = panel['TREAT_SIDO'] * panel['BLOCKBUSTER_MONTH']

# 로그 변환 예시 (FNB 매출 기준, 없는 컬럼이면 주석 처리)
if 'FNB' in panel.columns:
    panel['logVLM_FNB'] = np.log(panel['FNB'] + 1)
if 'CVS' in panel.columns:
    panel['logVLM_CVS'] = np.log(panel['CVS'] + 1)
if 'TRANS' in panel.columns:
    panel['logVLM_TRANS'] = np.log(panel['TRANS'] + 1)

# =========================================
# 5. 결과 저장
# =========================================

panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
print("패널 데이터 저장 완료:", output_panel)

print("\n패널 샘플 10행:")
print(panel.head(10))