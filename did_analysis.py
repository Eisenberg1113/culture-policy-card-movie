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
