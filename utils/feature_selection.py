import statsmodels.api as sm

def by_f_test(df, formula, repeat=10):
    result = None
    selected_ivs = []
    for i in range(repeat):
        print('{}번째 골라내는 중...'.format(i))
        model = sm.OLS.from_formula(formula, data=df)
        result = model.fit()
        anova = sm.stats.anova_lm(result, typ=2)
        selected_ivs = [iv[0] for iv in anova.iterrows() if iv[1][3] < 0.01]
        if len(selected_ivs) >= 0:
            formula = 'scale(price_doc) ~ ' + ' + '.join(selected_ivs)
        else:
            return result, selected_ivs
        print(result.rsquared, result.rsquared_adj, result.condition_number)
        print(len(selected_ivs), selected_ivs)
        # print(anova)
        print("="*100)
    return result, selected_ivs


def divide_numeric_cate(df, columns, drop_num, drop_cate):
    numeric = df._get_numeric_data().columns.values
    for n in drop_num:
        if n in numeric:
            numeric.drop(n)
    cate = list(set(df.columns) - set(numeric))
    for c in drop_cate:
        if c in cate:
            cate.drop(c)
    return numeric.drop(drop_num), cate.drop(drop_cate)
