# build_panel.py
import os
import pandas as pd
import numpy as np

# =========================================
# 0. 경로 설정 (레포 루트 기준)
# =========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

card_file    = os.path.join(DATA_DIR, "card_2024_2025_only.csv")
theater_file = os.path.join(DATA_DIR, "KC_497_DMSTC_MCST_THEART_2025.csv")  # cp949
movie_file   = os.path.join(DATA_DIR, "movie_monthly_cumulative.csv")       # utf-8-sig
output_panel = os.path.join(DATA_DIR, "panel_sido_month.csv")

# =========================================
# 1) 카드: GB2 → 분석 카테고리 매핑 후 시도×월 집계
# =========================================
chunksize = 500_000
card_agg_list = []

def map_cat_gb2(gb2: str) -> str:
    if pd.isna(gb2):
        return "OTHER"
    s = str(gb2)
    if ("외식" in s) or ("음식" in s) or ("식당" in s) or ("카페" in s):
        return "FNB"          # 식음료/외식
    if ("종합쇼핑" in s) or ("패션쇼핑" in s) or ("의류" in s) or ("패션" in s) or ("쇼핑" in s):
        return "SHOP"         # 쇼핑(종합+패션)
    if ("공연관람" in s) or ("문화" in s) or ("공연" in s) or ("영화" in s):
        return "CULTURE"      # 공연/문화(영화 포함 넓게)
    return "OTHER"

required_card_cols = ["TA_YM", "가맹점광역시도", "GB2", "VLM", "USEC"]
# 큰 파일이니 청크로 읽음
for chunk in pd.read_csv(card_file, chunksize=chunksize, encoding="utf-8-sig", engine="python"):
    missing = [c for c in required_card_cols if c not in chunk.columns]
    if missing:
        raise ValueError(f"카드 데이터에 필요한 컬럼이 없습니다: {missing}")

    # 기간 필터: 2024-01 ~ 2025-12
    chunk["TA_YM"] = chunk["TA_YM"].astype(str)
    chunk = chunk[(chunk["TA_YM"] >= "202401") & (chunk["TA_YM"] <= "202512")]
    if chunk.empty:
        continue

    chunk["SIDO_SHORT"] = chunk["가맹점광역시도"].astype(str)
    chunk["CAT_GRP"] = chunk["GB2"].apply(map_cat_gb2)

    grp = (chunk
           .groupby(["SIDO_SHORT", "TA_YM", "CAT_GRP"], as_index=False)
           .agg(VLM=("VLM", "sum"), USEC=("USEC", "sum")))
    card_agg_list.append(grp)

if not card_agg_list:
    raise ValueError("카드 데이터에서 기간 조건에 맞는 레코드가 없습니다. TA_YM/파일 경로를 확인하세요.")

card_agg = pd.concat(card_agg_list, ignore_index=True)
card_agg = (card_agg
            .groupby(["SIDO_SHORT", "TA_YM", "CAT_GRP"], as_index=False)
            .agg(VLM=("VLM", "sum"), USEC=("USEC", "sum")))

# 시도×월 피벗 (금액 VLM 기준)
card_pivot_vlm = (card_agg
                  .pivot_table(index=["SIDO_SHORT", "TA_YM"],
                               columns="CAT_GRP",
                               values="VLM",
                               aggfunc="sum",
                               fill_value=0)
                  .reset_index())
card_pivot_vlm.columns.name = None

# =========================================
# 2) 극장: 시도별 극장 수 집계 + Treat 더미
#    (파일 인코딩: cp949)
# =========================================
theater = pd.read_csv(theater_file, encoding="cp949")

if "sido_nm" not in theater.columns:
    raise ValueError("극장 데이터에 'sido_nm' 컬럼이 없습니다.")

def normalize_sido_long_to_short(x: str) -> str:
    if pd.isna(x): return x
    x = str(x)
    if x.endswith("특별시"):     return x[:-3]
    if x.endswith("광역시"):     return x[:-3]
    if x.endswith("특별자치시"): return x[:-5]
    if x.endswith("특별자치도"): return x[:-5]
    if x.endswith("도"):
        base = x[:-1]
        if len(base) >= 3 and base.endswith(("북", "남")):
            return base[-2:]   # 경상북 -> 경북
        return base
    return x

theater["SIDO_SHORT"] = theater["sido_nm"].apply(normalize_sido_long_to_short)

id_col = "id_poi" if "id_poi" in theater.columns else theater.columns[0]
gu_theater = (theater.groupby("SIDO_SHORT", as_index=False)
              .agg(THEATER_CNT=(id_col, "nunique")))

med_cnt = gu_theater["THEATER_CNT"].median()
gu_theater["TREAT_SIDO"] = (gu_theater["THEATER_CNT"] >= med_cnt).astype(int)

# =========================================
# 3) 영화: 월별 블록버스터 요약 (누적 관객 300만 기준)
#    입력: movie_monthly_cumulative.csv (월별 누적 관객/매출 포함)
#    기대 컬럼:
#      - MOVIE_NM, YM(YYYYMM), VIEWNG_NMPR_CO(해당월 관객),
#        cum_viewers(누적 관객), SALES_PRICE(해당월 매출), cum_sales(누적 매출),
#        IS_BLOCKBUSTER_MOVIE(누적≥300만이면 1), IS_CROSS_MONTH(해당월에 300만 돌파면 1)
# =========================================
movie = pd.read_csv(movie_file, encoding="utf-8-sig")

# 컬럼 이름 방어적으로 매핑
col_map = {}
for want in ["MOVIE_NM", "YM", "VIEWNG_NMPR_CO", "SALES_PRICE",
             "cum_viewers", "cum_sales",
             "IS_BLOCKBUSTER_MOVIE", "IS_CROSS_MONTH"]:
    if want not in movie.columns:
        # 대소문자/유사명 처리
        cand = [c for c in movie.columns if c.lower() == want.lower()]
        if cand:
            col_map[cand[0]] = want
movie = movie.rename(columns=col_map)

required_movie = ["MOVIE_NM", "YM", "IS_BLOCKBUSTER_MOVIE", "IS_CROSS_MONTH"]
missing_movie = [c for c in required_movie if c not in movie.columns]
if missing_movie:
    raise ValueError(f"영화 데이터에 필요한 컬럼이 없습니다: {missing_movie}")

# YM 정규화
movie["YM"] = movie["YM"].astype(str).str[:6]

# 월별 ‘300만 임계 돌파’ 횟수: IS_CROSS_MONTH의 합
cross_month = (movie.groupby("YM", as_index=False)
               .agg(BB_MOVIE_CNT_CROSS=("IS_CROSS_MONTH", "sum")))

# 월별 ‘블록버스터 존재’ 수(해당 월에 상영 중인 300만 달성 영화의 "제목" 기준 고유 개수)
bb_presence = (movie[movie["IS_BLOCKBUSTER_MOVIE"] == 1]
               .groupby(["YM", "MOVIE_NM"], as_index=False)
               .agg(any_flag=("IS_BLOCKBUSTER_MOVIE", "max"))
               .groupby("YM", as_index=False)
               .agg(BB_MOVIE_CNT_PRESENCE=("MOVIE_NM", "nunique")))

# 월별 합계(옵션): 관객·매출
sum_cols = {}
if "VIEWNG_NMPR_CO" in movie.columns:
    sum_cols["VIEWERS_SUM"] = ("VIEWNG_NMPR_CO", "sum")
if "SALES_PRICE" in movie.columns:
    sum_cols["SALES_SUM"] = ("SALES_PRICE", "sum")

month_sums = None
if sum_cols:
    month_sums = movie.groupby("YM", as_index=False).agg(**sum_cols)

# 월별 요약 병합
bb_month = cross_month.merge(bb_presence, on="YM", how="outer").fillna(0)
if month_sums is not None:
    bb_month = bb_month.merge(month_sums, on="YM", how="left").fillna(0)

bb_month["BLOCKBUSTER_MONTH"] = (bb_month["BB_MOVIE_CNT_PRESENCE"] > 0).astype(int)

# =========================================
# 4) 패널 조인 (카드 × 극장 × 블록버스터)
# =========================================
panel = card_pivot_vlm.copy()  # SIDO_SHORT, TA_YM, [CULTURE, FNB, SHOP, OTHER ...]

# 극장 조인
panel = panel.merge(gu_theater, on="SIDO_SHORT", how="left")

# 블록버스터 조인 (TA_YM ↔ YM)
panel = panel.merge(
    bb_month[["YM", "BB_MOVIE_CNT_CROSS", "BB_MOVIE_CNT_PRESENCE", "BLOCKBUSTER_MONTH",
              *(["VIEWERS_SUM", "SALES_SUM"] if month_sums is not None else [])]],
    left_on="TA_YM",
    right_on="YM",
    how="left"
)

# 결측 채움
for c in ["BB_MOVIE_CNT_CROSS", "BB_MOVIE_CNT_PRESENCE", "BLOCKBUSTER_MONTH"]:
    panel[c] = panel[c].fillna(0).astype(int)

# DID
panel["DID"] = panel["TREAT_SIDO"] * panel["BLOCKBUSTER_MONTH"]

# 로그 파생: FNB/SHOP/CULTURE 중 존재하는 것만 생성
if "FNB" in panel.columns:
    panel["logVLM_FNB"] = np.log(panel["FNB"] + 1)
if "SHOP" in panel.columns:
    panel["logVLM_SHOP"] = np.log(panel["SHOP"] + 1)
if "CULTURE" in panel.columns:
    panel["logVLM_CULTURE"] = np.log(panel["CULTURE"] + 1)

# 분석 전에 NaN 최소화
panel = panel.dropna(subset=["TREAT_SIDO", "BLOCKBUSTER_MONTH"])

# =========================================
# 5) 저장 + 간단 요약 출력
# =========================================
panel.to_csv(output_panel, index=False, encoding="utf-8-sig")
print("✅ 패널 저장 완료:", output_panel)

print("\n[샘플 8행]")
cols_show = [c for c in ["SIDO_SHORT","TA_YM","CULTURE","FNB","SHOP",
                         "THEATER_CNT","TREAT_SIDO",
                         "BB_MOVIE_CNT_CROSS","BB_MOVIE_CNT_PRESENCE","BLOCKBUSTER_MONTH",
                         "DID","logVLM_FNB","logVLM_SHOP","logVLM_CULTURE"] if c in panel.columns]
print(panel[cols_show].head(8))

print("\n[BLOCKBUSTER_MONTH × TREAT_SIDO 교차표]")
print(pd.crosstab(panel["BLOCKBUSTER_MONTH"], panel["TREAT_SIDO"]))

print("\n[월별 블록버스터 비율]")
print(panel.groupby("TA_YM")["BLOCKBUSTER_MONTH"].mean())
