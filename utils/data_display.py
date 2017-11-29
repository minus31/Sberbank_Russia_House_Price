import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import utils.preprocessor as pp



def first_look_at(df):
    for col in df.columns.drop('timestamp'):
        feature = df[col]
        if feature.dtype == np.int or feature.dtype == np.float: 
            value_counts = feature.value_counts()
            xmin = min(value_counts.index)
            xmax = max(value_counts.index)
            plt.subplot(221)
            plt.xlabel(col)
            plt.hist(df[col].dropna(), 100)
            
            log_feature = feature.apply(np.log).replace([np.inf, -np.inf], np.nan).dropna()
            if xmin <= 0:
                feature += xmin + 1
            plt.subplot(222)
            plt.xlabel("log({})".format(col))
            plt.hist(log_feature, 100)
            
            pair = df[[col, 'price_doc']]
            plt.subplot(223)
            plt.xlabel(col)
            plt.ylabel("log(price_doc)")
            sns.regplot(pair[col], pair['price_doc'].apply(np.log), line_kws={'color': 'red'})
            
            log_pair = df[[col, 'price_doc']].apply(np.log).dropna()
            if xmin <= 0:
                log_pair[col] += xmin + 1
            plt.subplot(224)
            plt.xlabel("log({})".format(col))
            plt.ylabel("log(price_doc)")
            sns.regplot(log_pair[col], log_pair['price_doc'], line_kws={'color': 'red'})
            
            corr, p = sp.stats.pearsonr(df[[col, 'price_doc']].dropna()[col], df[[col, 'price_doc']].dropna()['price_doc'])
            print(col, 'min: {}'.format(xmin), 'max: {}'.format(xmax), corr, p)
        elif df[col].dtype == np.object:     
            sns.violinplot(df[col], df['price_doc'])
            print(df[col].value_counts())

        plt.show()
        print('='*100)

def plot_missing_data(df):
    missing_df = pp.find_missing_data_columns(df)
    # if missing_df == None: return
    f, ax = plt.subplots(figsize=(6, 4))
    plt.xticks(rotation = '270')
    sns.barplot(x = missing_df.missing_column, y = (missing_df.missing_count/len(df))*100, color='#ff9999')
    ax.set(title = 'missing data by feature', ylabel = '% of missing data')
    plt.show()
    missing_df.shape