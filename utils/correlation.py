import numpy as np
import pandas as pd

def pick_highly_correlated_features(df, columns, min_corr):
    pairs = []
    for col in columns:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        corrs = [(col, c, abs(df[col].corr(df[c]))) for c in df.columns.values.tolist() if c != col]
        corrs.sort(key=lambda item: item[2], reverse=True)
        for item in corrs:
            if item[2] > min_corr:
                pairs.append(item)
            else:
                break
    return pd.DataFrame(pairs, columns=['missing_col', 'highest corr with', 'corr'])

def pick_highly_correlated_IVs(df, target_col, min_corr, min_unique_values = 0):
    if not np.issubdtype(df[target_col].dtype, np.number):
        Exception('{}은 numeric data가 아닙니다.'.format(target_col))
    # if len(df[col].value_counts().index) < min_unique_values:
    #     Exception('{}로 상관관계를 계산하기에는 유니크한 값이 너무 작습니다.'.format(col))

    corrs = []
    for col in df._get_numeric_data().drop(target_col, axis=1).columns:
        if len(df[col].value_counts().index) < min_unique_values: continue
        corr = abs(df[target_col].corr(df[col]))
        if corr > min_corr:
            corrs.append((col, corr))
    
    return corrs 