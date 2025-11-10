import os
import pandas as pd
import numpy as np

# =========================================
# 0. 파일 경로 설정 (레포 루트 기준)
# =========================================

# 이 스크립트(build_panel.py)가 있는 폴더 = 레포 루트라고 가정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일 경로들 (네가 말한 파일명으로 수정)
card_file    = os.path.join(DATA_DIR, "card_2024_2025_only.csv")         # 2024~25 카드데이터
theater_file = os.path.join(DATA_DIR, "KC_497_DMSTC_MCST_THEART_2025.csv")  # 극장 위치 데이터
movie_file   = os.path.join(DATA_DIR, "BOX_OFFIC_MOVIE_2024_2025_ALL.csv")  # 24~25 통합 영화·박스오피스
output_panel = os.path.join(DATA_DIR, "panel_sido_month.csv")               # 최종 분석용 패널

# =========================================
# 1. 카드데이터 전처리 + 시도×월×업종 집계
# =========================================

chunksize = 500_000
card_agg_list = []

# GB3 → 분석용 업종 그룹으로 매핑
def map_cat_gb3(gb3):
    """
    GB3 문자열을 받아서 분석용 카테고리로 매핑.
    실제 GB3 값 보고 여기 조건은 나중에 더 다듬어도 됨.
    """
    if pd.isna(gb3):
        return "OTHER"

    gb3 = str(gb3)

    if ("음식" in gb3) or ("외식" in gb3) or ("식당" in gb3) or ("카페" in gb3):
        return "FNB"       # Food & Beverage
    if "편의점" in gb3:
        return "CVS"       # 편의점
    if ("교통" in gb3) or ("주차" in gb3) or ("택시" in gb3):
        return "TRANS"     # 교통 관련
    if ("문화" in gb3) or ("공연" in gb3) or ("영화" in gb3):
        return "CULTURE"   # 문화/공연
    # 스포츠/여행은 카드 전처리 단계에서 이미 제거된 상태라고 가정
    return "OTHER"

for chunk in pd.read_csv(
    card_file,
    chunksize=chunksize,
    encoding="utf-8-sig",
    engine="python"
):
    # 연월(TA_YM)을 문자열로 맞추기
    if "TA_YM" not in chunk.columns:
        raise ValueError("카드 데이터에 'TA_YM' 컬럼이 없습니다. 컬럼명을 확인하세요.")

    chunk["TA_YM"] = chunk["TA_YM"].astype(str)

    # 안전하게 2024-01 ~ 2025-12만 사용 (card_2024_2025_only라 거의 다 맞을 거지만 한 번 더 필터)
    chunk = chunk[(chunk["TA_YM"] >= "202401") & (chunk["TA_YM"] <= "202512")]

    if chunk.empty:
        continue

    # 시도 컬럼명: '가맹점광역시도' 사용
    if "가맹점광역시도" not in chunk.columns:
        raise ValueError("카드 데이터에 '가맹점광역시도' 컬럼이 없습니다. 실제 컬럼명을 확인하세요.")

    chunk["가맹점광역시도"] = chunk["가맹점광역시도"].astype(str)

    # GB3 → 분석용 카테고리
    if "GB3" not in chunk.columns:
        raise ValueError("카드 데이터에 'GB3' 컬럼이 없습니다.")
    chunk["CAT_GRP"] = chunk["GB3"].apply(map_cat_gb3)

    # 기본 집계 (시도×월×카테고리 기준 매출/건수 합계)
    grp = (
        chunk
        .groupby(["가맹점광역시도", "TA_YM", "CAT_GRP"], as_index=False)
        .agg(
            VLM=("VLM", "sum"),
            USEC=("USEC", "sum")
        )
    )
    card_agg_list.append(grp)

# 여러 청크 합치기
card_agg = pd.concat(card_agg_list, ignore_index=True)

# 혹시 중복 행 있으면 다시 한 번 집계해서 정리
card_agg = (
    card_agg
    .groupby(["가맹점광역시도", "TA_YM", "CAT_GRP"], as_index=False)
    .agg(
        VLM=("VLM", "sum"),
        USEC=("USEC", "sum")
    )
)

# 피벗: 시도×월 기준으로 FNB/CVS/TRANS/CULTURE/OTHER 컬럼 생성
card_pivot = card_agg.pivot_table(
    index=["가맹점광역시도", "TA_YM"],
    columns="CAT_GRP",
    values="VLM",
    aggfunc="sum",
    fill_value=0
).reset_index()

card_pivot.columns.name = None   # 컬럼 MultiIndex 해제

# =========================================
# 2. 극장 데이터 → 시도별 극장 수 / Treat 더미
# =========================================

theater = pd.read_csv(theater_file, encoding="utf-8-sig")

if "sido_nm" not in theater.columns:
    raise ValueError("극장 데이터에 'sido_nm' 컬럼이 없습니다.")

def normalize_sido_long_to_short(x: str) -> str:
    """
    '서울특별시' -> '서울', '경상북도' -> '경북' 형태로 줄이기
    (카드데이터 '가맹점광역시도' == '서울','경북' 형식이라고 가정)
    """
    if pd.isna(x):
        return x
    x = str(x)
    if x.endswith("특별시"):
        return x[:-3]
    if x.endswith("광역시"):
        return x[:-3]
    if x.endswith("특별자치시"):
        return x[:-5]
    if x.endswith("특별자치도"):
        return x[:-5]
    if x.endswith("도"):
        base = x[:-1]   # '경상북도' -> '경상북'
        if len(base) >= 3 and base.endswith(("북", "남")):
            return base[-2:]  # '경상북' -> '경북'
        return base
    return x

theater["SIDO_SHORT"] = theater["sido_nm"].apply(normalize_sido_long_to_short)

# 시도별 극장 수
if "id_poi" not in theater.columns:
    raise ValueError("극장 데이터에 'id_poi' 컬럼이 없습니다. 극장 ID 컬럼명을 확인하세요.")

gu_theater = (
    theater
    .groupby("SIDO_SHORT", as_index=False)
    .agg(THEATER_CNT=("id_poi", "nunique"))
)

# 극장 수 중앙값 기준 Treat 시도(1) vs Control 시도(0)
med_cnt = gu_theater["THEATER_CNT"].median()
gu_theater["TREAT_SIDO"] = (gu_theater["THEATER_CNT"] >= med_cnt).astype(int)

# =========================================
# 3. 영화/박스오피스 → 블록버스터 개봉월 플래그
# =========================================

movie = pd.read_csv(movie_file, encoding="utf-8-sig")

# 필수 컬럼 체크
for col in ["OPN_DE", "TOT_SCRN_CO"]:
    if col not in movie.columns:
        raise ValueError(f"영화 데이터에 '{col}' 컬럼이 없습니다.")

# 개봉월 (YYYYMM) 추출
movie["OPN_DE"] = movie["OPN_DE"].astype(str)
movie["OPN_YM"] = movie["OPN_DE"].str.slice(0, 6)

# TOT_SCRN_CO 숫자로 변환 (혹시 문자열이면)
movie["TOT_SCRN_CO"] = pd.to_numeric(movie["TOT_SCRN_CO"], errors="coerce")

# 블록버스터 기준 (스크린 수 상위 20%)
thr = movie["TOT_SCRN_CO"].quantile(0.8)

movie["IS_BLOCKBUSTER"] = (movie["TOT_SCRN_CO"] >= thr).astype(int)

# 월별 블록버스터 개수/유무
bb_month = (
    movie
    .groupby("OPN_YM", as_index=False)
    .agg(
        BB_MOVIE_CNT=("IS_BLOCKBUSTER", "sum")
    )
)
bb_month["BLOCKBUSTER_MONTH"] = (bb_month["BB_MOVIE_CNT"] > 0).astype(int)

# =========================================
# 4. 시도×월 패널 조인 (카드 + 극장 + 블록버스터)
# =========================================

panel = card_pivot.copy()

# 시도 이름을 SIDO_SHORT로 맞추기
panel = panel.rename(columns={"가맹점광역시도": "SIDO_SHORT"})

# 극장 정보 조인 (SIDO_SHORT 기준)
panel = panel.merge(gu_theater, on="SIDO_SHORT", how="left")

# 블록버스터 월 조인 (TA_YM ↔ OPN_YM)
panel = panel.merge(
    bb_month[["OPN_YM", "BB_MOVIE_CNT", "BLOCKBUSTER_MONTH"]],
    left_on="TA_YM",
    right_on="OPN_YM",
    how="left"
)

# 블록버스터 정보 없는 월은 0으로 채우기
panel["BB_MOVIE_CNT"] = panel["BB_MOVIE_CNT"].fillna(0).astype(int)
panel["BLOCKBUSTER_MONTH"] = panel["BLOCKBUSTER_MONTH"].fillna(0).astype(int)

# DID 상호작용 변수
panel["DID"] = panel["TREAT_SIDO"] * panel["BLOCKBUSTER_MONTH"]

# 로그 변환 예시 (해당 컬럼 있을 때만)
if "FNB" in panel.columns:
    panel["logVLM_FNB"] = np.log(panel["FNB"] + 1)
if "CVS" in panel.columns:
    panel["logVLM_CVS"] = np.log(panel["CVS"] + 1)
if "TRANS" in panel.columns:
    panel["logVLM_TRANS"] = np.log(panel["TRANS"] + 1)
if "CULTURE" in panel.columns:
    panel["logVLM_CULTURE"] = np.log(panel["CULTURE"] + 1)

# =========================================
# 5. 결과 저장
# =========================================

panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
print("패널 데이터 저장 완료:", output_panel)

print("\n패널 샘플 10행:")
print(panel.head(10))
