"""
A little package that I made to make my work with data a bit easier.
It is mainly made up of little functions that are useful for exploring data,
    but are not part of any library (that I know of)
Enjoy! :)
"""

"""
simple_reg_model - Calculates r2 score, Root mean square error and cross-validates the data set
with Root mean square error scoring. Also plots a scatter plot with real and predicted labels.

Parameters:
    model - sklearn ML model, Machine learning model that you want to get the results of
    df - pandas.DataFrame object, machine learning model features (X)
    target - pandas.Series object, machine learning model labels (y)
    return_model - boolean value (default=False), if True, the function will return the fit model

Note:
    This is a function used for regression models.
"""
def simple_reg_model(model, df, target, return_model=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    try:
        df = df.drop(target.name, axis=1)
    except:
        pass

    model.fit(df, target)
    predictions = model.predict(df)

    r2 = r2_score(target, predictions)
    print('R2 score: ', r2)

    mse = mean_squared_error(target, predictions)
    rmse = np.sqrt(mse)
    print('RMSE: ', rmse)

    print('Cross-Validation: ')
    model_scores = cross_val_score(model, df, target,
                                      scoring='neg_mean_squared_error', cv=4)
    model_rmse_scores = np.sqrt(-model_scores)
    print('\tScores:', model_rmse_scores)
    print('\tMean:', model_rmse_scores.mean())
    print('\tStandard deviation:', model_rmse_scores.std())

    plt.xlabel('Real Labels')
    plt.ylabel('Predictions')
    plt.title('Labels vs Predictions')
    plt.scatter(target, predictions, color='black', s=5)

    if return_model==True:
        return model

"""
plot_label_corr - Plots a scatter plot for better visualization of correlation between every feature in
given data set and labels, which are to predict.

Parameters:
    df - pandas.DataFrame object, machine learning model features (X)
    target - pandas.Series object, machine learning model labels (y)
    size - number variable, scatter plot points' size

Note:
    Also used to easily visualize outliers in a data set.
    Reduce the "size" parameter if the data is too dense and the plot is unclear.
"""
def plot_label_corr(df, target, size):
    import pandas as pd
    import matplotlib.pyplot as plt

    for feature in list(df):
        plt.scatter(target, df[feature], color='black', s=size)

        plt.xlabel('Target')
        plt.ylabel(feature)
        plt.show()

"""
scale_nums - Scales all the number features in a given data set using scikit-learns' Standard Scaler.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on

Note:
    Poor generalization.
"""
def scale_nums(df):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_to_scale = []
    for feature in list(df):
        if np.issubdtype(df[feature].dtype, np.number) == True:
                features_to_scale.append(feature)
    for feature in features_to_scale:
        df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    return df

"""
get_abv_corr - Return passed data frame but only keep the columns that correlate to the target above certain
threshold.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    target - pandas.Series object, values to correlate features with
    threshold - number value, drop values that correlate to the target below this threshold(default=0)

Note:
    Somewhat primitive, might drop features that could be important (random forest feature weights could,
    for an example find some features useful).
"""
def get_abv_corr(df, target, threshold=0):
    import pandas as pd

    ftr_list = []
    for feature, value in df.corrwith(target).sort_values(ascending=False).iteritems():
        if abs(value) >= threshold:
            ftr_list.append(feature)

    return ftr_list

"""
print_abv_corr - Similar to the function above, but only prints the features and their values above certain
threshold.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    target - pandas.Series object, values to correlate features with
    threshold - number value, drop values that correlate to the target below this threshold (default 0)

Note:
    /
"""
def print_abv_corr(df, target, threshold=0, round_value=4):
    import pandas as pd

    for feature, value in df.corrwith(target).sort_values(ascending=False).iteritems():
        if abs(value) >= threshold:
            print(feature, round(value, round_value))

"""
print_abv_ft_corr - Similar to the function above, goes through every feature.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    threshold_down - number value, drop values that correlate to the target below this threshold
    threshold_up - number value, drop values that correlate to the target above this threshold

Note:
    /
"""
def print_abv_ft_corr(df, threshold_down=0, threshold_up=1, round_value=4):
    import pandas as pd

    for ftr in list(df):
        printed = False
        for feature, value in df.corrwith(df[ftr]).iteritems():
            if (threshold_up >= abs(value) >= threshold_down) & (feature != ftr):
                if printed == False:
                    print(ftr)
                    printed = True
                    print(feature, round(value, round_value))
                else:
                    print(feature, round(value, round_value))

"""
degrade_columns - Used for exctracting the binary state value from the feature that is compiled of two
or more features. This way, the data that would have been useless in "dual" feature, is assigned to
"singular" feature, thus reducing the number of features and improving the correlation between the
"singular" feature and the target. This function was made for Szeged Weather data set and it was used
to remove "dual" features:
    (Windy_and_sunny=1 ==> Windy=1, Sunny=1)
In App Store dataset was further tweaked and expanded from n=2 number of features in "dual" feature
to be able to support any number of features in "dual" feature. Filltering, feature list and debugging
were also added.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    sep - string type variable, string slice used for separating two features' names
    filter - bloolean value, drop features which have a dominate value (0,1)
    filter_threshold - decimal value, threshold below which 0 or 1 must be for it's feature
    to be dropped (default=0.0456)
    return_ftr_list - boolean value, if True returns a list containing newly added features
    (default=False)
    debug - boolean value, if True prints added and removed features while the process is done
    for better insigts (default=False)

Note:
    GOLD.
"""

def degrade_columns(df, sep, filter=False, filter_threshold=0.0456, return_ftr_list=False,
                    debug=False):
    import numpy as np
    import pandas as pd

    new_features = []
    for column in list(df):
        if sep in column:
            cur_list = column.split(sep)
            for col in cur_list:
                if col not in df.columns:
                    df[col] = 0
                    new_features.append(col)
                    if debug == True:
                        print('Added ',col)
                df.loc[df[column].eq(1), col] = 1
            df = df.drop(column, axis=1)

    if filter == True:
        for feature in new_features:
            if (df.groupby(feature)[feature].count()/len(df)).iloc[0] == 1:
                df = df.drop(feature, axis=1)
                if debug == True:
                    print('Dropped', feature)

        new_features = [x for x in new_features if x in list(df)]

        for feature in new_features:
            if ((df.groupby(feature)[feature].count()/len(df))[0] <= filter_threshold)|(
            (df.groupby(feature)[feature].count()/len(df))[1] <= filter_threshold):
                df = df.drop(feature, axis=1)
                if debug == True:
                    print('Dropped ', feature)
            else:
                if debug == True:
                    print(feature, ' not dropped.')

        new_features = [x for x in new_features if x in list(df)]

    if return_ftr_list==True:
        return df, new_features
    else:
        return df

"""
simple_class_model - Displaying a simple classification model, prints CV accuracy scores, confusion matrix,
precision score, recall score and f1 score.

Parameters:
    model - sklearn ML model, Machine learning model that you want to get the results of
    df - pandas.DataFrame object, machine learning model features (X)
    target - pandas.Series object, machine learning model labels (y)
    return_model - boolean value (default=False), if True, the function will return the fit model

Note:
    This is a function used for classification models.
"""
def simple_class_model(model, df, target, return_model=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import precision_recall_curve

    try:
        df = df.drop(target.name, axis=1)
    except:
        pass

    model.fit(df, target)

    print('CV accuracy scores:')
    print(cross_val_score(model, df, target, cv=5, scoring="accuracy"))

    target_pred = cross_val_predict(model, df, target, cv=10)
    print('Confusion matrix:')
    print(confusion_matrix(target, target_pred))

    print('Precision score:')
    print(precision_score(target, target_pred))
    print('Recall score:')
    print(recall_score(target, target_pred))
    print('F1 score:')
    print(f1_score(target, target_pred))

    precision, recall, threshold = precision_recall_curve(target, target_pred)
    plt.plot(precision, recall)

    if return_model==True:
        return model

"""
one_hot_encode - One hot encodes given columns in the given data set, drops old columns

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    features - list objects, list of column names that will be encoded

Note:
    /
"""
def one_hot_encode(df, features):
    import pandas as pd

    dummy_df = pd.get_dummies(df[features])
    df = pd.concat([df.drop(features, axis=1), dummy_df], axis=1, sort=False)

    return df

"""
#OR USE IMBLEARN
under_sample - Excellent for unbalanced data sets with binary classes that
classification is performed on. Samples the false outcome part of the df
to be approximately the same size as the true part of the df, so the results are
less biased.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    target - string, binary target column of the data set
    scale - number object, size of false df (df.loc[df[target] == 0]) with
    respect to true df (default=1)
    bootstrap - boolean value, will random sample from false df be bootstrapped
    (default=False)

Note:
    The false df might end up smaller than true df if bootstrap is turned on,
    depends on the how many times is false df bigger than true df. If it is
    smaller after many draws, it is a good idea to increase the scale
    parameter.
"""
def under_sample(df, target, scale=1, bootstrap=False):
    import numpy as np
    import pandas as pd

    true_df = df.loc[df[target] == 1]
    false_ix = df.loc[df[target] == 0].index
    sample_size = int(round(scale * len(df.loc[df[target] == 1])))
    false_df = df.iloc[np.random.choice(false_ix, sample_size, replace=bootstrap)]
    undersampled_df = pd.concat([true_df, false_df])
    return undersampled_df

"""
rmsle - Returns root mean squared logarithmic error.

Parameters:
    target - pd.Series, label column
    predicted - pd.Series, predicted label column
    return_value - boolean value, return the value and store it in a given variable (default=False)

Note:
    /
"""
def rmsle(target, predicted, return_value=False):
    import numpy as np
    import pandas as pd

    sum = 0.0
    for x in range(len(predicted)):
        if predicted[x] < 0 or target[x] < 0:
            continue
        p = np.log(predicted[x]+1)
        r = np.log(target[x]+1)
        sum = sum + (p - r)**2

    if return_value == True:
        return (sum / len(predicted)) ** 0.5
    else:
        print((sum / len(predicted)) ** 0.5)

"""
plot_data_vs_pred - performs PCA on the whole data to be able to plot real vs. predicted values in a scatter
plot.

Parameters:
    df - pandas.DataFrame object, machine learning model features (X)
    target - pd.Series, label column
    predicted - pd.Series, predicted label column

Note:
    Data has to be scaled.
"""
def plot_data_vs_pred(df, target, predicted):
    import pandas as pd
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=1)
    pca_df = pd.DataFrame(pca.fit_transform(df)).reset_index(drop=True)
    plt.scatter(x=pca_df,
                y=target,
                c='yellow',
                s=10)
    plt.scatter(x=pca_df,
                y=predicted,
                c='black',
                s=8)
    plt.show()

"""
check_forest_corr - Checks feature importance based on xgb random forest weights and returns the list with
features which score is lower than the given threshold.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    target - pd.Series, label column
    objective - string, XGBoost objective
    return_list - boolean value, if true returns the list of features that are below given threshold (default=False)
    model - 'Regressor' or 'Classifier', denotes the type of XGBoost forest (default='Classifier')
    threshold - decimal value (range 0-1), threshold above which all correlated features will be added
    (0.001 def.)

Note:
    Not always optional.
"""
def check_forest_corr(df, target, objective, return_list=False, model='Classifier', threshold=0.001, n_jobs=-1):
    from xgboost import XGBClassifier, XGBRegressor
    import pandas as pd

    if return_list == True:
        drop_list = []

    if model == 'Classifier':
        forest = XGBClassifier(objective=objective, n_jobs=n_jobs)
    elif model == 'Regressor':
        forest = XGBRegressor(objective=objective, n_jobs=n_jobs)
    else:
        raise InputError('Invalid model type inputed.')
    forest.fit(df, target)

    for score, feature in sorted(zip(forest.feature_importances_, list(df)), reverse=True):
        if score>threshold:
            print(feature, round(score, 4))
        else:
            if return_list == True:
                drop_list.append(feature)

    if return_list == True:
        return drop_list

"""
drop_opposite - Drops very highly correlated column (for example if we have "Does Have" and "Does not Have"
feature, it will remove one of them because it is irrelevant).

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    threshold - decimal value (range 0-1), threshold above which all correlated features will be dropped
    (0.85 def.)

Note:
    /
"""
def drop_opposite(df, threshold=0.85):
    import pandas as pd

    curr_list = list(df)
    for ftr in list(df):
        if ftr in curr_list:
            for feature, value in df.corrwith(df[ftr]).iteritems():
                if (abs(value)>=threshold) & (feature != ftr):
                    df = df.drop(feature, axis=1)
        curr_list = list(df)
    return df

"""
merge_similar - Merges highly correlated columns (above given threshold) into one feature.

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    threshold - decimal value (range 0-1), threshold above which all correlated features will be merged
    (0.45 def.)

Note:
    Might loose column name information while performing this function for multiple times.
"""
def merge_similar(df, threshold=0.45):
    from sklearn.decomposition import PCA
    import pandas as pd

    pca = PCA(n_components=1)

    curr_list = list(df)
    curr_fts = []
    for ftr in list(df):
        if ftr in curr_list:
            for feature, value in df.corrwith(df[ftr]).iteritems():
                if (abs(value)>=threshold) & (feature != ftr):
                    curr_fts.append(feature)

        if len(curr_fts) != 0:
            curr_fts.append(ftr)
            curr_fts_df = df[curr_fts]
            new_feature_name = '{}+{}'.format(ftr,str(len(curr_fts)-1))
            df = df.drop(curr_fts, axis=1)
            df[new_feature_name] = pca.fit_transform(curr_fts_df)

        curr_list = list(df)
        curr_fts = []
    return df

"""
impute_lin_reg - Imputes missing values in pandas column based on given feature(s) that is(are) relevant to
the column with missing values using linear regression. Returns imputed pandas.Series .

Parameters:
    df - pandas.DataFrame object, the df that the transformation is performed on
    related_features - list of strings, column names in the data set
    impute_col - string, the column who's values are being imputed

Note:
    Might introduce bias.
"""
def impute_lin_reg(df, related_features, impute_col):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import pandas as pd

    pca = PCA(n_components=1)

    data = df[related_features]
    data = pd.DataFrame(pca.fit_transform(data), columns=['Feature'])
    data['Target'] = df[impute_col]

    X = pd.DataFrame(data.loc[~np.isnan(data['Target']), 'Feature'])
    y = pd.DataFrame(data.loc[~np.isnan(data['Target']), 'Target'])

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    X_predict = pd.DataFrame(data.loc[np.isnan(data['Target']), 'Feature'])
    impute_values = lin_reg.predict(X_predict).tolist()
    impute_values = [num for sublist in impute_values for num in sublist]

    data.Target = data.Target.fillna(pd.Series(impute_values, index=X_predict.index))

    new_impute_col = data['Target']
    return new_impute_col

"""
look - Gives a good summary of the data columns.

Parameters:
    df - pandas.DataFrame object, the df we want to look up
    pred - string value, column of the df which is prediction target (default=None)

Note:
    It is derived from Kaggle Notebook on Housing Competition and modified.
    Credits go to: https://www.kaggle.com/mgmarques
"""
def look(df, pred=None):
    import numpy as np
    import pandas as pd

    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = round((df.isnull().sum()/obs)*100, 4)
    skewness = df.skew()

    if pred is None:
        cols = ['Types', 'Counts', 'Distincts', 'Nulls', 'Missing ratio (%)', 'Uniques', 'Skewness']
        mtx = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness], axis=1, sort=False)

    else:
        corr = round(df.corr()[pred], 5)
        corr_col = 'Corr '+pred
        cols = ['Types', 'Counts', 'Distincts', 'Nulls', 'Missing_ratio (%)', 'Uniques', 'Skewness', corr_col]
        mtx = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness, corr],
                        axis=1, sort=False)

    mtx.columns = cols
    return mtx

"""
plot_spread - Plots the histogram (with KDE) and the boxplot of specified columns within a dataframe.

Parameters:
    df - pandas.DataFrame or pandas.Series object, the df/series we want to look up
    cols - single string of list object, names of columns we want to examine

Note:
    /
"""
def plot_spread(df, cols=0, color='b'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(cols, list):
        pass
    else:
        cols = [cols]

    if isinstance(df, pd.DataFrame):
        for column in cols:
            plt.figure(figsize=(16,4))

            if df[column].nunique() < 60:
                plt.subplot(1, 2, 1)
                sns.distplot(df[column], color=color, kde_kws={'bw':1})
                plt.title('Histogram')
            else:
                plt.subplot(1, 2, 1)
                sns.distplot(df[column], color=color)
                plt.title('Histogram')

            plt.subplot(1, 2, 2)
            sns.boxplot(df[column])
            plt.title('Boxplot')

            plt.show()
    else:
        plt.figure(figsize=(16,4))

        if df.nunique() < 60:
            plt.subplot(1, 2, 1)
            sns.distplot(df, color=color, kde_kws={'bw':1})
            plt.title('Histogram')
        else:
            plt.subplot(1, 2, 1)
            sns.distplot(df, color=color)
            plt.title('Histogram')

        plt.subplot(1, 2, 2)
        sns.boxplot(df)
        plt.title('Boxplot')

        plt.show()

"""
group_by - Groups unique values in dataframe and prints the frequency of every one. Limited by the
number of unique values within dataframe columns or by specified columns.

Parameters:
    df - pandas.DataFrame object, the df we want to look up
    nuniques - number object, maximum number of unique values a column can posses in order to be
    listed (default=999999)
    specific_features - list or string type, name or list of names of desired columns

Note:
    /
"""
def group_by(df, specific_features=None, nuniques=999999):
    import numpy as np
    import pandas as pd

    if specific_features == None:
        for feature in list(df):
            if df[feature].nunique() <= nuniques:
                print((df.groupby(feature)[feature].count()/len(df)).sort_values(ascending=False))

    elif (specific_features != None) and (nuniques != 999999):
        raise ValueError('Wrong inputs!')

    else:
        if isinstance(specific_features, list):
            pass
        else:
            specific_features = [specific_features]

        for feature in specific_features:
            print((df.groupby(feature)[feature].count()/len(df)).sort_values(ascending=False))