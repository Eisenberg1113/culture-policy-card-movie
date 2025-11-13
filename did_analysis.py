# did_analysis.py
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# -----------------------------
# 0) 경로/입력
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PANEL_PATH = os.path.join(DATA_DIR, "panel_sido_month.csv")

panel = pd.read_csv(PANEL_PATH, dtype={"TA_YM": str})

# -----------------------------
# 1) 기본 QA
# -----------------------------
print(panel.shape, panel.columns.tolist())
print(panel.isna().mean().sort_values(ascending=False).head(10))
print(panel[['SIDO_SHORT', 'TA_YM']].duplicated().sum(), "중복 rows")

# 정렬(지역-월)
panel = panel.sort_values(['SIDO_SHORT', 'TA_YM']).reset_index(drop=True)

# -----------------------------
# 2) 로그타깃 보강(없으면 생성)
# -----------------------------
for col in ['FNB', 'SHOP', 'CULTURE']:
    if col in panel.columns and f'logVLM_{col}' not in panel.columns:
        panel[f'logVLM_{col}'] = np.log(panel[col].clip(lower=0) + 1)

# DID 기본 변수(없으면 생성)
if 'DID' not in panel.columns:
    if {'TREAT_SIDO', 'BLOCKBUSTER_MONTH'}.issubset(panel.columns):
        panel['DID'] = panel['TREAT_SIDO'] * panel['BLOCKBUSTER_MONTH']
    else:
        raise ValueError("DID 생성에 필요한 TREAT_SIDO/BLOCKBUSTER_MONTH가 없습니다.")

# 월별 블록버스터 비율(로그용)
if 'BLOCKBUSTER_MONTH' in panel.columns:
    bb_rate = panel.groupby('TA_YM')['BLOCKBUSTER_MONTH'].mean()
    print("\n[월별 블록버스터 비율]\n", bb_rate)

# -----------------------------
# 3) 강도형(연속형) 충격 변수 구성
# -----------------------------
if 'VIEWERS_SUM' not in panel.columns:
    print("[WARN] VIEWERS_SUM이 없어 0으로 채운 변수를 생성합니다.")
    panel['VIEWERS_SUM'] = 0.0

panel['BB_INTENSITY'] = panel['VIEWERS_SUM'].fillna(0.0).astype(float)

# 분산 0 방지(표준화)
std_ = panel['BB_INTENSITY'].std(ddof=0)
if std_ == 0 or np.isnan(std_):
    panel['BB_INTENSITY_Z'] = 0.0
else:
    panel['BB_INTENSITY_Z'] = (panel['BB_INTENSITY'] - panel['BB_INTENSITY'].mean()) / std_
panel['BB_INTENSITY_Z'] = panel['BB_INTENSITY_Z'].clip(-4, 4)

# 강도형 상호작용
panel['DID_INT'] = panel['TREAT_SIDO'] * panel['BB_INTENSITY_Z']

# -----------------------------
# 4) 시간 인덱스 & 지역별 트렌드, 리드/래그
# -----------------------------
month_order = sorted(panel['TA_YM'].astype(str).unique())
month_to_int = {m: i for i, m in enumerate(month_order)}
panel['t_idx'] = panel['TA_YM'].map(month_to_int).astype('int64')

panel['trend_by_sido'] = (
    panel.groupby('SIDO_SHORT', observed=False)['t_idx']
         .transform(lambda s: s - s.mean())
         .astype('float64')
)

panel['BB_INT_LAG1']  = panel.groupby('SIDO_SHORT', observed=False)['BB_INTENSITY_Z'].shift(1)
panel['BB_INT_LEAD1'] = panel.groupby('SIDO_SHORT', observed=False)['BB_INTENSITY_Z'].shift(-1)
panel['DID_LAG1']  = panel['TREAT_SIDO'] * panel['BB_INT_LAG1']
panel['DID_LEAD1'] = panel['TREAT_SIDO'] * panel['BB_INT_LEAD1']

# -----------------------------
# 5) 고정효과(FE)용 카테고리 지정
# -----------------------------
panel['SIDO_SHORT'] = panel['SIDO_SHORT'].astype('category')
panel['TA_YM_FE']   = panel['TA_YM'].astype('category')

# 분석 핵심변수 결측 제거(최소선)
need_cols = ['SIDO_SHORT', 'TA_YM', 'TREAT_SIDO', 'BLOCKBUSTER_MONTH', 'logVLM_FNB']
panel = panel.dropna(subset=[c for c in need_cols if c in panel.columns]).copy()

# -----------------------------
# 6) 회귀들
# -----------------------------
def pct(x):
    return (np.exp(x) - 1.0) * 100.0

def tidy_result(model, name):
    out = pd.DataFrame({
        'coef': model.params,
        'se': model.bse,
        'z_or_t': model.tvalues,
        'pval': model.pvalues
    })
    out.index.name = 'term'
    out.reset_index(inplace=True)
    out['model'] = name
    return out

tidy_all = []

# (A) 이산형 DID (비교용) — 클러스터(SE=SIDO)
m_bin = smf.ols("logVLM_FNB ~ DID + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel)\
          .fit(cov_type='cluster', cov_kwds={'groups': panel['SIDO_SHORT']})
print("\n== (A) Binary DID on FNB ==")
print(m_bin.summary())
if 'DID' in m_bin.params:
    print("해석(%) ≈", pct(m_bin.params['DID']))
tidy_all.append(tidy_result(m_bin, 'A_binary_FNB'))

# (B) 강도형 DID (연속 충격) — HC3
m_int = smf.ols("logVLM_FNB ~ DID_INT + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel)\
          .fit(cov_type='HC3')
print("\n== (B) Intensity DID (Z) on FNB ==")
print(m_int.summary())
if 'DID_INT' in m_int.params:
    print("해석(1σ 증가 시 %) ≈", pct(m_int.params['DID_INT']))
tidy_all.append(tidy_result(m_int, 'B_intensity_FNB'))

# (C) 리드/래그
m_es = smf.ols(
    "logVLM_FNB ~ DID_INT + DID_LAG1 + DID_LEAD1 + C(SIDO_SHORT) + C(TA_YM_FE)",
    data=panel
).fit(cov_type='HC3')
print("\n== (C) Intensity DID with lead/lag on FNB ==")
print(m_es.summary())
tidy_all.append(tidy_result(m_es, 'C_intensity_leadlag_FNB'))

# (D) 지역별 선형 트렌드 보강
m_trend = smf.ols(
    "logVLM_FNB ~ DID_INT + trend_by_sido + C(SIDO_SHORT) + C(TA_YM_FE)",
    data=panel
).fit(cov_type='HC3')
print("\n== (D) Intensity DID + region-specific linear trends on FNB ==")
print(m_trend.summary())
tidy_all.append(tidy_result(m_trend, 'D_intensity_trend_FNB'))

# (E) 이질적 효과: 극장수 중앙값 split
if 'THEATER_CNT' in panel.columns:
    q50 = panel['THEATER_CNT'].median()
    panel['HIGH_THEATER'] = (panel['THEATER_CNT'] >= q50).astype(int)
    panel['DID_INT_HIGH'] = panel['DID_INT'] * panel['HIGH_THEATER']
    m_het = smf.ols(
        "logVLM_FNB ~ DID_INT + DID_INT_HIGH + C(SIDO_SHORT) + C(TA_YM_FE)",
        data=panel
    ).fit(cov_type='HC3')
    print("\n== (E) Heterogeneity by theater density (median split) ==")
    print(m_het.summary())
    tidy_all.append(tidy_result(m_het, 'E_intensity_hetero_FNB'))

# (F) 다른 타깃: SHOP / CULTURE
for y in [c for c in ['logVLM_SHOP', 'logVLM_CULTURE'] if c in panel.columns]:
    mod = smf.ols(f"{y} ~ DID_INT + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel)\
             .fit(cov_type='HC3')
    print(f"\n== Intensity DID on {y} ==")
    print(mod.summary())
    if 'DID_INT' in mod.params:
        print("해석(1σ 증가 시 %) ≈", pct(mod.params['DID_INT']))
    tidy_all.append(tidy_result(mod, f'intensity_{y}'))

# =========================
# (X1) 리드/래그 확장 이벤트 스터디
# =========================
max_lag = 2
max_lead = 2

if 'BB_INTENSITY_Z' in panel.columns:
    for L in range(1, max_lag+1):
        panel[f'BB_INT_LAG{L}']  = panel.groupby('SIDO_SHORT', observed=False)['BB_INTENSITY_Z'].shift(L)
        panel[f'DID_LAG{L}']     = panel['TREAT_SIDO'] * panel[f'BB_INT_LAG{L}']
    for L in range(1, max_lead+1):
        panel[f'BB_INT_LEAD{L}'] = panel.groupby('SIDO_SHORT', observed=False)['BB_INTENSITY_Z'].shift(-L)
        panel[f'DID_LEAD{L}']    = panel['TREAT_SIDO'] * panel[f'BB_INT_LEAD{L}']

    leads = " + ".join([f"DID_LEAD{L}" for L in range(1, max_lead+1)])
    lags  = " + ".join([f"DID_LAG{L}"  for L in range(1, max_lag+1)])
    rhs   = "DID_INT"
    if leads: rhs += " + " + leads
    if lags:  rhs += " + " + lags

    m_es2 = smf.ols(
        f"logVLM_FNB ~ {rhs} + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel
    ).fit(cov_type='HC3')
    print("\n== (X1) Extended event-study on FNB ==")
    print(m_es2.summary())

# =========================
# (X2) 월 단위 클러스터로 강건성 확인
# =========================
m_month_cluster = smf.ols(
    "logVLM_FNB ~ DID_INT + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['TA_YM_FE']})
print("\n== (X2) FNB with month-clustered SEs ==")
print(m_month_cluster.summary())

# =========================
# (X3) 블록 부트스트랩(by SIDO) p-value
# =========================
import numpy as np
rng = np.random.default_rng(123)
def block_boot_pvalue(formula, data, group, B=999, target='DID_INT'):
    base = smf.ols(formula, data=data).fit()
    coef0 = base.params.get(target, np.nan)
    if np.isnan(coef0): 
        return np.nan, coef0
    groups = data[group].unique().tolist()
    boot_coefs = []
    for _ in range(B):
        samp_g = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([data.index[data[group]==g] for g in samp_g])
        boot = smf.ols(formula, data=data.loc[idx]).fit()
        boot_coefs.append(boot.params.get(target, np.nan))
    boot_coefs = np.array(boot_coefs)
    # 양측 p-value
    p = (np.abs(boot_coefs - boot_coefs.mean()) >= np.abs(coef0 - boot_coefs.mean())).mean()
    return float(p), float(coef0)

p_boot, bcoef = block_boot_pvalue(
    "logVLM_FNB ~ DID_INT + C(SIDO_SHORT) + C(TA_YM_FE)", panel, 'SIDO_SHORT', B=999, target='DID_INT'
)
print(f"\n== (X3) Block bootstrap p (by SIDO) for DID_INT on FNB ==\ncoef={bcoef:.4f}, p≈{p_boot:.3f}")

# =========================
# (X4) 연속 상호작용(극장수)
# =========================
if 'THEATER_CNT' in panel.columns:
    m_ix = smf.ols(
        "logVLM_FNB ~ DID_INT*THEATER_CNT + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel
    ).fit(cov_type='HC3')
    print("\n== (X4) FNB with DID_INT x THEATER_CNT (continuous interaction) ==")
    print(m_ix.summary())

# =========================
# (X5) 누적 충격(윈도우 합)
# =========================
if 'BB_INTENSITY_Z' in panel.columns:
    panel['BB_INT_Z_roll3'] = panel.groupby('SIDO_SHORT', observed=False)['BB_INTENSITY_Z']\
                                   .transform(lambda s: s.rolling(3, min_periods=1).sum())
    panel['DID_INT_roll3'] = panel['TREAT_SIDO'] * panel['BB_INT_Z_roll3']
    m_roll = smf.ols(
        "logVLM_FNB ~ DID_INT_roll3 + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel
    ).fit(cov_type='HC3')
    print("\n== (X5) FNB with 3-month cumulative intensity ==")
    print(m_roll.summary())

# =========================
# (X6) 플라시보(처리 한 달 당김)
# =========================
panel['BB_PLACEBO'] = panel.groupby('SIDO_SHORT', observed=False)['BLOCKBUSTER_MONTH'].shift(-1).fillna(0)
panel['DID_PLACEBO'] = panel['TREAT_SIDO'] * panel['BB_PLACEBO']
m_placebo = smf.ols(
    "logVLM_FNB ~ DID_PLACEBO + C(SIDO_SHORT) + C(TA_YM_FE)", data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['SIDO_SHORT']})
print("\n== (X6) Placebo (advance treatment by 1 month) ==")
print(m_placebo.summary())

# =========================
# (X7) 결과 요약 저장 보강
# =========================
def save_params(name, res, target):
    try:
        row = {
            'model': name,
            'coef': res.params.get(target, np.nan),
            'se': res.bse.get(target, np.nan),
            'pvalue': res.pvalues.get(target, np.nan)
        }
    except Exception:
        row = {'model': name, 'coef': np.nan, 'se': np.nan, 'pvalue': np.nan}
    return row

summ_rows = []
summ_rows.append(save_params('FNB_HC3', m_int, 'DID_INT'))
summ_rows.append(save_params('FNB_monthCluster', m_month_cluster, 'DID_INT'))
summ_rows.append({'model':'FNB_bootstrap_p', 'coef': bcoef, 'se': np.nan, 'pvalue': p_boot})
if 'm_ix' in locals():
    summ_rows.append(save_params('FNB_interaction_THEATER', m_ix, 'DID_INT'))
if 'm_roll' in locals():
    summ_rows.append(save_params('FNB_roll3', m_roll, 'DID_INT_roll3'))
summ_df = pd.DataFrame(summ_rows)
out2 = os.path.join(DATA_DIR, "did_results_summary_v2.csv")
summ_df.to_csv(out2, index=False, encoding="utf-8-sig")
print("[Saved]", out2)

# -----------------------------
# 7) 결과물 저장(락 + 요약표)
# -----------------------------
out_panel = os.path.join(DATA_DIR, "panel_sido_month_v2_locked.csv")
panel.to_csv(out_panel, index=False, encoding="utf-8-sig")
print("\n[Saved]", out_panel)

if tidy_all:
    tidy_df = pd.concat(tidy_all, ignore_index=True)
    out_reg = os.path.join(DATA_DIR, "did_results_summary.csv")
    tidy_df.to_csv(out_reg, index=False, encoding="utf-8-sig")
    print("[Saved]", out_reg)
