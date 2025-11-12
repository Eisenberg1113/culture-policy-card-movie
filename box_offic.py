import os
import re
import glob
import pandas as pd
import numpy as np

# -----------------------------
# 0) 경로/입출력 설정
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 이 파일이 레포 루트에 있을 때
DATA_DIR = os.path.join(BASE_DIR, "data")

# 월별 파일이 2024/, 2025/ 폴더에 있다고 가정 (패턴은 필요시 수정)
PATTERNS = [
    os.path.join(DATA_DIR, "2024", "BOX_OFFIC_MOVIE_2024*.csv"),
    os.path.join(DATA_DIR, "2025", "BOX_OFFIC_MOVIE_2025*.csv"),
]

OUT_MOVIE_ALL = os.path.join(DATA_DIR, "movie_monthly_cumulative.csv")
OUT_BB_MONTH  = os.path.join(DATA_DIR, "bb_month_summary.csv")

# 블록버스터 임계치 (누적 관객)
BLOCKBUSTER_VIEWERS_THRESHOLD = 3_000_000

# -----------------------------
# 1) 유틸: 파일명에서 YYYYMM 추출
# -----------------------------
ym_pat = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")

def extract_ym_from_path(path: str) -> str:
    """
    파일 경로에서 YYYYMM을 찾아 문자열로 반환. 없으면 None.
    """
    m = ym_pat.search(os.path.basename(path))
    return m.group(0) if m else None

def to_int_safe(x):
    """
    쉼표/공백/NaN 섞인 숫자를 안전하게 정수로 변환.
    """
    if pd.isna(x):
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    # 1,234,567 같은 포맷 정리
    s = s.replace(",", "")
    try:
        return int(float(s))
    except:
        return 0

# -----------------------------
# 2) 월별 파일 읽어서 하나로 합치기 (cp949)
# -----------------------------
files = []
for p in PATTERNS:
    files.extend(glob.glob(p))
files = sorted(files)

if not files:
    raise RuntimeError("월별 영화 CSV를 찾지 못했습니다. DATA_DIR/2024, 2025 안의 파일 패턴을 확인하세요.")

dfs = []
for f in files:
    ym = extract_ym_from_path(f)
    if ym is None:
        print(f"[경고] 파일명에서 YYYYMM을 찾지 못함: {f} (스킵)")
        continue

    # cp949로 시도 (네 환경 기준)
    df = pd.read_csv(f, encoding="cp949", engine="python")

    # 필요한 컬럼만 골라두기 (존재 여부 확인 후 유연하게 처리)
    needed_cols = [
        "MOVIE_NM",         # 영화명
        "OPN_DE",           # 개봉일 (YYYYMMDD)
        "TOT_SCRN_CO",      # 스크린 수
        "SALES_PRICE",      # 해당월 매출
        "VIEWNG_NMPR_CO",   # 해당월 관객
        "NLTY_NM",          # 국적 (선택)
        "GENRE_NM",         # 장르 (선택)
    ]
    exist_cols = [c for c in needed_cols if c in df.columns]
    df = df[exist_cols].copy()

    # 월 정보 컬럼 추가
    df["YM"] = ym

    # 숫자 컬럼 안전 변환
    if "SALES_PRICE" in df.columns:
        df["SALES_PRICE"] = df["SALES_PRICE"].apply(to_int_safe)
    if "VIEWNG_NMPR_CO" in df.columns:
        df["VIEWNG_NMPR_CO"] = df["VIEWNG_NMPR_CO"].apply(to_int_safe)
    if "TOT_SCRN_CO" in df.columns:
        df["TOT_SCRN_CO"] = pd.to_numeric(df["TOT_SCRN_CO"], errors="coerce").fillna(0).astype(int)

    # 문자열 정리
    if "MOVIE_NM" in df.columns:
        df["MOVIE_NM"] = df["MOVIE_NM"].astype(str).str.strip()
    if "OPN_DE" in df.columns:
        df["OPN_DE"] = df["OPN_DE"].astype(str).str.replace(r"\D", "", regex=True)

    dfs.append(df)

movie_m = pd.concat(dfs, ignore_index=True)

# -----------------------------
# 3) 영화별 정렬 → 누적 관객/매출 계산
# -----------------------------
# YM 정렬을 위해 문자열 기준으로 정렬 (YYYYMM 형식이므로 문자열 정렬 = 시간 정렬)
movie_m = movie_m.sort_values(["MOVIE_NM", "YM"]).reset_index(drop=True)

# 누적합
movie_m["cum_viewers"] = movie_m.groupby("MOVIE_NM")["VIEWNG_NMPR_CO"].cumsum() if "VIEWNG_NMPR_CO" in movie_m.columns else 0
movie_m["cum_sales"]   = movie_m.groupby("MOVIE_NM")["SALES_PRICE"].cumsum()    if "SALES_PRICE"    in movie_m.columns else 0

# 영화별 총합/최종값
agg_total = movie_m.groupby("MOVIE_NM").agg(
    total_viewers = ("cum_viewers", "max"),
    total_sales   = ("cum_sales",   "max"),
    max_screens   = ("TOT_SCRN_CO", "max") if "TOT_SCRN_CO" in movie_m.columns else ("YM", "count")
).reset_index()

# 블록버스터 영화 여부: 총 누적 관객 기준
agg_total["IS_BLOCKBUSTER_MOVIE"] = (agg_total["total_viewers"] >= BLOCKBUSTER_VIEWERS_THRESHOLD).astype(int)

# 최초 300만 돌파 월 (없으면 NaN)
first_cross = (
    movie_m.loc[movie_m["cum_viewers"] >= BLOCKBUSTER_VIEWERS_THRESHOLD, ["MOVIE_NM", "YM"]]
    .groupby("MOVIE_NM", as_index=False)
    .agg(CROSS_YM=("YM", "min"))
)
agg_total = agg_total.merge(first_cross, on="MOVIE_NM", how="left")

# 원본에 합치기
movie_all = movie_m.merge(agg_total, on="MOVIE_NM", how="left")

# 이 달이 “최초 돌파 달”인가?
movie_all["IS_CROSS_MONTH"] = (movie_all["YM"] == movie_all["CROSS_YM"]).astype(int)

# -----------------------------
# 4) 월별 요약 (정책/상권 연계에 쓰기 쉬운 형태)
# -----------------------------
bb_month = (
    movie_all.groupby("YM", as_index=False)
    .agg(
        BB_MOVIE_CNT_CROSS    = ("IS_CROSS_MONTH", "sum"),           # 임계 '돌파'가 발생한 영화 수
        BB_MOVIE_CNT_PRESENCE = ("IS_BLOCKBUSTER_MOVIE", "sum"),     # 블록버스터로 분류된 영화들이 '존재'한 수(중복 카운트)
        VIEWERS_SUM           = ("VIEWNG_NMPR_CO", "sum"),
        SALES_SUM             = ("SALES_PRICE", "sum")
    )
    .sort_values("YM")
)

# -----------------------------
# 5) 저장 (UTF-8-sig)
# -----------------------------
movie_all.to_csv(OUT_MOVIE_ALL, index=False, encoding="utf-8-sig")
bb_month.to_csv(OUT_BB_MONTH,  index=False, encoding="utf-8-sig")

print("✅ 저장 완료")
print(" - 영화 월별 누적 파일:", OUT_MOVIE_ALL)
print(" - 블록버스터 월 요약:", OUT_BB_MONTH)

print("\n[영화 월별 누적 샘플]")

print(movie_all.head(10)[[
    "MOVIE_NM","YM","VIEWNG_NMPR_CO","cum_viewers",
    "SALES_PRICE","cum_sales","IS_BLOCKBUSTER_MOVIE","CROSS_YM","IS_CROSS_MONTH"
]])

cols = ["MOVIE_NM","YM","VIEWNG_NMPR_CO","cum_viewers",
        "SALES_PRICE","cum_sales","IS_BLOCKBUSTER_MOVIE",
        "CROSS_YM","IS_CROSS_MONTH"]

cross_hits = (
    movie_all.loc[movie_all["IS_CROSS_MONTH"] == 1, cols]
              .sort_values(["CROSS_YM","MOVIE_NM"])
)
print(cross_hits.head(20))


print("\n[월별 블록버스터 요약 샘플]")
print(bb_month.head(24))
