import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

panel = pd.read_csv("data/panel_sido_month.csv", dtype={"TA_YM": str})

# 기본 QA
print(panel.shape, panel.columns.tolist())
print(panel.isna().mean().sort_values(ascending=False).head(10))
print(panel[['SIDO_SHORT','TA_YM']].duplicated().sum(), "중복 rows")

# 월 순서 정렬
panel = panel.sort_values(['SIDO_SHORT','TA_YM']).reset_index(drop=True)

# 로그 타깃 보강(혹시 없으면 생성)
for col in ['FNB','SHOP','CULTURE']:
    if col in panel.columns and f'logVLM_{col}' not in panel.columns:
        panel[f'logVLM_{col}'] = np.log(panel[col].clip(lower=0) + 1)

# 분석에 꼭 필요한 열 결측 제거
need = ['SIDO_SHORT','TA_YM','TREAT_SIDO','BLOCKBUSTER_MONTH','logVLM_FNB']
panel = panel.dropna(subset=[c for c in need if c in panel.columns]).copy()

print(pd.crosstab(panel['BLOCKBUSTER_MONTH'], panel['TREAT_SIDO']))
print(panel.groupby('TA_YM')['BLOCKBUSTER_MONTH'].mean().rename('BB_rate_by_month'))


# 카테고리형으로 명시
panel['SIDO_SHORT'] = panel['SIDO_SHORT'].astype('category')
panel['TA_YM'] = panel['TA_YM'].astype('category')

# 기본 DID: logVLM_FNB ~ DID + 지역FE + 월FE
m1 = smf.ols(
    formula="logVLM_FNB ~ DID + C(SIDO_SHORT) + C(TA_YM)",
    data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['SIDO_SHORT']})
print(m1.summary())

did_pct = np.exp(m1.params['DID']) - 1
print("DID 효과(%) ≈", did_pct*100)

targets = [c for c in ['logVLM_SHOP','logVLM_CULTURE'] if c in panel.columns]
for y in targets:
    m = smf.ols(formula=f"{y} ~ DID + C(SIDO_SHORT) + C(TA_YM)",
                data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['SIDO_SHORT']})
    print(f"\n== {y} 결과 ==")
    print(m.summary().tables[1])

# 동태 효과 계산
panel['TA_YM_num'] = panel['TA_YM'].astype(int)
bb_set = set(panel.loc[panel['BLOCKBUSTER_MONTH']==1,'TA_YM_num'])

panel['BB_LAG1']  = panel['TA_YM_num'].apply(lambda x: 1 if (x-1) in bb_set else 0)
panel['BB_LEAD1'] = panel['TA_YM_num'].apply(lambda x: 1 if (x+1) in bb_set else 0)

for var in ['BB_LAG1','BB_LEAD1']:
    panel[f'DID_{var}'] = panel['TREAT_SIDO'] * panel[var]

m_es = smf.ols("logVLM_FNB ~ DID + DID_BB_LAG1 + DID_BB_LEAD1 + C(SIDO_SHORT) + C(TA_YM)",
               data=panel).fit(cov_type='cluster', cov_kwds={'groups': panel['SIDO_SHORT']})
print(m_es.summary().tables[1])

# 강건성 체크(서울/경기 제외)
mask = ~panel['SIDO_SHORT'].isin(['서울','경기'])
m2 = smf.ols("logVLM_FNB ~ DID + C(SIDO_SHORT) + C(TA_YM)", data=panel[mask])\
       .fit(cov_type='cluster', cov_kwds={'groups': panel[mask]['SIDO_SHORT']})
print(m2.summary().tables[1])

# (A) 밸런스 테이블: 평균(표준편차)
cols = ['logVLM_FNB','logVLM_SHOP','logVLM_CULTURE','THEATER_CNT']
bal = (panel.groupby('TREAT_SIDO')[cols]
       .agg(['mean','std','min','max','median']).T)
print("\n[Balance table]\n", bal)

# (B) 전후(또는 블록버스터 유무) 평균 비교
pre = panel[panel['BLOCKBUSTER_MONTH']==0]
post = panel[panel['BLOCKBUSTER_MONTH']==1]
for c in cols:
    print(f"{c}: pre={pre[c].mean():.3f}, post={post[c].mean():.3f}")

# 연속형 충격 z-score
panel['BB_INTENSITY'] = panel['VIEWERS_SUM'].fillna(0)
panel['BB_INTENSITY_Z'] = (panel['BB_INTENSITY'] - panel['BB_INTENSITY'].mean()) / panel['BB_INTENSITY'].std()

# 상호작용
panel['DID_INT'] = panel['TREAT_SIDO'] * panel['BB_INTENSITY_Z']

import statsmodels.formula.api as smf

# 두 방향 고정효과(TWFE) + 견고표준오차(HC3 또는 cluster)
fe_formula = "logVLM_FNB ~ DID_INT + C(SIDO_SHORT) + C(TA_YM)"
m_int = smf.ols(fe_formula, data=panel).fit(cov_type='HC3')
print(m_int.summary())
print("해석(%)≈", (np.exp(m_int.params['DID_INT'])-1)*100)

# 리드·래그를 위해 월 인덱스
panel = panel.copy()
panel['t'] = panel.groupby([]).ngroup()  # 간단히 쓰려면 아래 보조 인덱스 방식:
# 더 명확히:
ta_map = {ym:i for i, ym in enumerate(sorted(panel['TA_YM'].unique()))}
panel['t_idx'] = panel['TA_YM'].map(ta_map)

# intensity의 시차/선행 (±1)
def lag(df, col, k):
    return df.sort_values('t_idx').groupby('SIDO_SHORT')[col].shift(k)

panel['BB_INT_LAG1']  = panel.groupby('SIDO_SHORT')['BB_INTENSITY_Z'].shift(1)
panel['BB_INT_LEAD1'] = panel.groupby('SIDO_SHORT')['BB_INTENSITY_Z'].shift(-1)

panel['DID_LAG1']  = panel['TREAT_SIDO']*panel['BB_INT_LAG1']
panel['DID_LEAD1'] = panel['TREAT_SIDO']*panel['BB_INT_LEAD1']

es_formula = "logVLM_FNB ~ DID_INT + DID_LAG1 + DID_LEAD1 + C(SIDO_SHORT) + C(TA_YM)"
m_es = smf.ols(es_formula, data=panel).fit(cov_type='HC3')
print(m_es.summary())

panel['trend_by_sido'] = panel.groupby('SIDO_SHORT')['t_idx'].transform(lambda s: (s - s.mean()))
rob_formula = "logVLM_FNB ~ DID_INT + trend_by_sido + C(SIDO_SHORT) + C(TA_YM)"
m_trend = smf.ols(rob_formula, data=panel).fit(cov_type='HC3')
print(m_trend.summary())

q50 = panel['THEATER_CNT'].median()
panel['HIGH_THEATER'] = (panel['THEATER_CNT'] >= q50).astype(int)
panel['DID_INT_HIGH'] = panel['DID_INT'] * panel['HIGH_THEATER']
het_formula = "logVLM_FNB ~ DID_INT + DID_INT_HIGH + C(SIDO_SHORT) + C(TA_YM)"
print(smf.ols(het_formula, data=panel).fit(cov_type='HC3').summary())

# 분석에 쓰지 않는 결측행 제거(이미 최소선은 처리했지만 한 번 더)
keep = panel.dropna(subset=['logVLM_FNB','BB_INTENSITY'])
keep.to_csv(os.path.join(DATA_DIR,"panel_sido_month_v2_locked.csv"), index=False, encoding="utf-8-sig")


