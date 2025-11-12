import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

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


