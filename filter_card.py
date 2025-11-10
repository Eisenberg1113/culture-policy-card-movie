import pandas as pd

base_dir = r"c:\Users\Bang\Desktop\공공데이터활용 대회(영화 정보)"

infile  = base_dir + r"\card_filtered.csv"              # 2023+ / 스포츠·여행 제거된 파일
outfile = base_dir + r"\card_2024_2025_only.csv"        # 분석용(24–25년만)

chunksize = 500_000
first = True

for chunk in pd.read_csv(
    infile,
    chunksize=chunksize,
    encoding="utf-8-sig",
    engine="python"
):
    # TA_YM을 문자열로 맞추고
    if 'TA_YM' not in chunk.columns:
        raise ValueError("TA_YM 컬럼이 없습니다. 카드데이터 컬럼명을 다시 확인하세요.")
    chunk['TA_YM'] = chunk['TA_YM'].astype(str)

    # 2024-01 ~ 2025-12만 필터
    sub = chunk[(chunk['TA_YM'] >= '202401') & (chunk['TA_YM'] <= '202512')]

    if sub.empty:
        continue

    sub.to_csv(
        outfile,
        index=False,
        mode='w' if first else 'a',
        header=first,
        encoding="utf-8-sig"
    )
    first = False

print("24–25년 카드데이터 저장 완료:", outfile)
