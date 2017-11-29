import numpy as np
import pandas as pd
import utils.correlation as corr
import statsmodels.api as sm
import utils.statsmodels_helper as smh
from sklearn.preprocessing import OneHotEncoder

def merge(df_1, df_2, on_col):
    df_tm = pd.merge(df_1, df_2, on=[on_col, on_col])
    df_tm_cols = df_tm.columns.tolist()
    df_tm_cols = df_tm_cols[:290] + df_tm_cols[291:] + [df_tm_cols[290]]
    df_tm = df_tm[df_tm_cols]
    return df_tm

def clean_column_names(cols):
    cleaned_cols = [col.replace('-', '').replace('+', '').replace(':', '') for col in cols]
    cleaned_cols = ['c_' + col if col[0].isdigit() else col for col in cleaned_cols]
    return cleaned_cols

def clean_data(df):
    # build_year 1500이전 nan으로
    df.loc[df.build_year < 1500, 'build_year'] = np.nan
    df.loc[df.build_year > 2016, 'build_year'] = np.nan
    
    # floor가 0이면 nan으로
    df.loc[df.floor==0, 'floor'] = np.nan

    # max_floor가 0이면 nan으로
    df.loc[df.max_floor==0, 'max_floor'] = np.nan

    # max_floor가 floor보다 크면 nan으로
    df.loc[df.floor>df.max_floor, 'max_floor'] = np.nan

    # full_sq, life_sq가 0이면 nan으로
    df.loc[df.full_sq==0, 'full_sq'] = np.nan
    df.loc[df.life_sq==0, 'life_sq'] = np.nan

    # full_sq가 life_sq보다 작으면 nan으로
    df.loc[df.life_sq>df.full_sq, 'life_sq'] = np.nan

    # kitch_sq가 life_sq보다 크면 nan으로
    df.loc[df.kitch_sq>df.life_sq, 'kitch_sq'] = np.nan
    
    df.loc[df.state == 33, 'state'] = 3
    
    df.loc[df.num_room < 0, 'num_room'] = np.nan

    df['material'].fillna(0, inplace=True)
        
    # 이상한 숫자값들 45,34 ...
    if 'modern_education_share' in df: del df['modern_education_share']
    if 'old_education_build_share' in df: del df['old_education_build_share']
    if 'child_on_acc_pre_school' in df: del df['child_on_acc_pre_school']


    consts = [col for col in df.columns if len(df[col].value_counts().index) == 1]
    for const in consts:
        del df[const]

    df = df.replace(['no data'], ['nodata'])


    # 뉴머릭한 카테고리컬 독립변수들인데 유니크값이 너무 많아서 없앤다.
    del df['ID_railroad_station_walk']
    del df['ID_railroad_station_avto']
    del df['ID_big_road1']
    del df['ID_big_road2']
    del df['ID_railroad_terminal']
    del df['ID_bus_terminal']
    del df['ID_metro']

    # 50% 이상 미싱 데이터가 있으면 없애버린다
    del df['provision_retail_space_sqm']
    del df['theaters_viewers_per_1000_cap']
    del df['museum_visitis_per_100_cap']

    # too many dummy variables
    del df['sub_area']

    # material은 카테고리
    # df['material'] = df['material'].astype(np.str, copy=False)
    df['material'] = df['material'].replace([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ['a', 'b', 'c', 'd', 'e', 'f', 'e'])
    return df

def categorize(df):
    df['material'] = df['material'].astype(np.object, copy=False)

def find_missing_data_columns(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['missing_column', 'missing_count']
    missing_df = missing_df.loc[missing_df['missing_count'] > 0]
    return missing_df

def impute_by_regressoin(df, repeat):
    for _ in range(repeat):
        missing_df = find_missing_data_columns(df)
        for col in missing_df['missing_column']:    
            if not np.issubdtype(df[col].dtype, np.number): continue
            correlated_pairs = corr.pick_highly_correlated_IVs(df, col, 0.3, 2)
            correlated_ivs = list(map(lambda x: x[0], correlated_pairs))
            correlated_df = df[correlated_ivs + [col]]
            # print(correlated_df)
            formula = smh.make_statsmodels_ols_formula(list(map(lambda col: col[0], correlated_pairs)), [], col, True)
            print(formula)
            model = sm.OLS.from_formula(formula, data=correlated_df.dropna(subset=correlated_ivs))
            result = model.fit()
            result.summary()
            predicted = result.predict(correlated_df)
            std = np.sqrt(np.var(result.resid))
            predicted = predicted.apply(lambda x: x + np.random.normal(loc=0, scale=std))

            df = pd.merge(df, predicted.to_frame('predicted'), left_index=True, right_index=True)
            df[col].fillna(df['predicted'], inplace=True)
            del df['predicted']

def impute_by_regressoin2(df, repeat, corr_thresh):
    for i in range(repeat):
        print('{}번째 반복중...'.format(i+1))
        pairs = []
        missing_df = find_missing_data_columns(df)
        for missing_col in missing_df['missing_column']:
            if not np.issubdtype(df[missing_col].dtype, np.number): continue
            corrs = [ (missing_col, c, abs(df[missing_col].corr(df[c]))) for c in df._get_numeric_data().columns.drop('price_doc') if c != missing_col]
            corrs.sort(key=lambda item: item[2], reverse=True)
            for item in corrs:
                if item[2] > corr_thresh:
                    pairs.append(item)
                else:
                    break
        df_nan_col_with_highly_correlated_col = pd.DataFrame(pairs, columns=['missing_col', 'highest corr with', 'corr'])
        for row in df_nan_col_with_highly_correlated_col.iterrows():
            if df[row[1][0]].isnull().sum() <= 0: continue
            nan_col = row[1][0]
            high_corr_col = row[1][1]
            corr = row[1][2]
            print('{}를 {}와 imputing 중...'.format(nan_col, high_corr_col))
                
            df_temp = pd.DataFrame(df[[high_corr_col, nan_col]], columns=[high_corr_col, nan_col])
            df_temp = df_temp.dropna()
            
            df_temp = sm.add_constant(df_temp)
            X = df_temp.values[:, :2]
            y = df_temp.values[:, 2]
            result = sm.OLS(y, X).fit()
        #     print(result.summary())
            
            dfX = sm.add_constant(df[high_corr_col])
            predicted = result.predict(dfX)
            
            df = pd.merge(df, predicted.to_frame('predicted'), left_index=True, right_index=True)
            df[nan_col].fillna(df['predicted'], inplace=True)
            del df['predicted']
    return df

def impute_by_regression3(df, df_test, repeat, corr_thresh):
    for i in range(repeat):
        print('{}번째 반복중...'.format(i+1))
        pairs = []
        missing_df = find_missing_data_columns(df)
        for missing_col in missing_df['missing_column']:
            if not np.issubdtype(df[missing_col].dtype, np.number): continue
            corrs = [ (missing_col, c, abs(df[missing_col].corr(df[c]))) for c in df._get_numeric_data().columns.drop('price_doc') if c != missing_col]
            corrs.sort(key=lambda item: item[2], reverse=True)
            for item in corrs:
                if item[2] > corr_thresh:
                    pairs.append(item)
                else:
                    break
        df_nan_col_with_highly_correlated_col = pd.DataFrame(pairs, columns=['missing_col', 'highest corr with', 'corr'])
        for row in df_nan_col_with_highly_correlated_col.iterrows():
            if df[row[1][0]].isnull().sum() <= 0: continue
            nan_col = row[1][0]
            high_corr_col = row[1][1]
            corr = row[1][2]
            print('{}를 {}와 imputing 중...'.format(nan_col, high_corr_col))
                
            df_temp = pd.DataFrame(df[[high_corr_col, nan_col]], columns=[high_corr_col, nan_col])
            df_temp = df_temp.dropna()
            
            df_temp = sm.add_constant(df_temp)
            X = df_temp.values[:, :2]
            y = df_temp.values[:, 2]
            result = sm.OLS(y, X).fit()
        #     print(result.summary())
            
            dfX = sm.add_constant(df[high_corr_col])
            predicted = result.predict(dfX)
            
            dfX_test = sm.add_constant(df_test[high_corr_col])
            predicted_test = result.predict(dfX_test)

            df = pd.merge(df, predicted.to_frame('predicted'), left_index=True, right_index=True)
            df[nan_col].fillna(df['predicted'], inplace=True)
            del df['predicted']

            df_test = pd.merge(df_test, predicted_test.to_frame('predicted'), left_index=True, right_index=True)
            df_test[nan_col].fillna(df_test['predicted'], inplace=True)
            del df_test['predicted']
    return df, df_test

def apply_log(df, numeric_cols):
    for col in numeric_cols:
        min_val = min(df[col].value_counts().index)
        if min_val < 0:
            df[col] -= min_val
            df[col] += 1
        else:
            df[col] += 1
    df[numeric_cols].apply(np.log)

def scale_up_positive(df, numeric_cols):
    for col in numeric_cols:
        min_val = min(df[col].value_counts().index)
        if min_val < 0:
            df[col] -= min_val
            df[col] += 1
        else:
            df[col] += 1



def remove_outliers(df, formula, repeat=1):
    result = None
    for i in range(repeat):
        print('{}번 반복중... 아웃라이어 찾는중...'.format(i+1))
        model = sm.OLS.from_formula(formula, data=df)
        result = model.fit()
        influence = result.get_influence()
        distances, pvalues = influence.cooks_distance
        threshold = 4/(len(distances) - len(df.columns.drop(['timestamp', 'price_doc']))-1)
        outliers = [idx for idx, d in enumerate(distances) if d > threshold]
        print("아웃라이어는 {}개 입니다.".format(len(outliers)))
        df.drop(df.index[outliers], inplace=True)
    return model, result
