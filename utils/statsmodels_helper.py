import statsmodels as sm
import utils.str_helper as sh

# numeric_ivs = df_t._get_numeric_data().columns.drop('price_doc')
# cate_ivs = list(set(df_tm.columns.drop('timestamp')) - set(df_tm._get_numeric_data().columns))
# sh.make_statsmodels_ols_formula(df_t._get_numeric_data().columns.drop('price_doc'), cate_ivs, 'price_doc', True)


def make_statsmodels_ols_formula(numeric_ivs, categorical_ivs, dv, log_ivs=[], degree=1, scale=True):
    polynomials = []
    if degree > 1:
        for i in range(2, degree + 1):
            if scale:
                polynomials = list(map(lambda iv: 'scale(I({}**{}))'.format(iv, i), numeric_ivs))
            else:
                polynomials = list(map(lambda iv: 'I({}**{})'.format(iv, i), numeric_ivs))

    if len(log_ivs) > 0:
        numeric_ivs = ["np.log({})".format(iv) if iv in log_ivs else iv for iv in numeric_ivs ]
    if scale:
        numeric_ivs = ["scale({})".format(iv) if scale else iv for iv in numeric_ivs ]

    formula = ''
    if scale:
        if dv in log_ivs:
            formula = 'scale(np.log({})) ~ '.format(dv)
        else:
            formula = 'scale({}) ~ '.format(dv)
    else:
        if dv in log_ivs:
            formula = 'np.log({}) ~ '.format(dv)
        else:
            formula = '{} ~ '.format(dv)

    if len(categorical_ivs) > 0:
        if len(numeric_ivs) > 0:
            formula += " + ".join(list(map(lambda iv: 'C({})'.format(iv), categorical_ivs)))
        else:
            formula += " + ".join(list(map(lambda iv: 'C({})-1'.format(iv), categorical_ivs)))
    
    if len(polynomials) > 0:
        return formula + " + " + " + ".join(numeric_ivs) + " + " + " + ".join(polynomials)
    else:
        return formula + " + " + " + ".join(numeric_ivs)



def unwrap(ivs):
    cate = [sh.find_between_r(f, 'C(', ')') for f in ivs]
    num = [sh.find_between_r(f, 'scale(', ')') for f in ivs]
    return list(filter(None, cate + num))
