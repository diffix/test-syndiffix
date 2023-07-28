import os
import sys
import pandas as pd
import json
import time
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import testUtils
import fire
import pprint
import sdmTools

pp = pprint.PrettyPrinter(indent=4)

''' This is used in SLURM to run one features measure
'''

CATEGORY_THRESHOLD = 15


def loadCsv(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''], low_memory=False)


def isCategorical(column):
    return column.dtype == 'object' or column.nunique() <= CATEGORY_THRESHOLD


def getFeatureTypes(df):
    ordinal_features = []
    continuous_features = []
    categorical_features = []

    for colname in df.columns:
        column = df[colname]
        nunique = column.nunique()
        if nunique <= CATEGORY_THRESHOLD:
            categorical_features.append(colname)
        elif column.dtype == 'object':
            ordinal_features.append(colname)
        else:
            continuous_features.append(colname)

    return ordinal_features, continuous_features, categorical_features


def oneHotFeatureNames(encoder, categorical_features):
    cats = [
        encoder._compute_transformed_categories(i)
        for i, _ in enumerate(encoder.categories_)
    ]

    feature_names = []
    inverse_names = {}

    for i in range(len(cats)):
        category = categorical_features[i]
        names = [category + "$$" + str(label) for label in cats[i]]
        feature_names.extend(names)

        for name in names:
            inverse_names[name] = category

    return feature_names, inverse_names


def preprocess(df, oneHotEncode=True, varianceThreshold=True):
    ordinal_features, continuous_features, categorical_features = getFeatureTypes(df)

    if not oneHotEncode:
        ordinal_features += categorical_features
        categorical_features = []

    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(), ordinal_features),
            ('num', RobustScaler(), continuous_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features),
        ],
        sparse_threshold=0)

    result = preprocessor.fit_transform(df)

    inverse_lookup = {
        name: name for name in ordinal_features + continuous_features
    }

    categorical_names = []

    if len(categorical_features) > 0:
        categorical_names, categorical_inverse = oneHotFeatureNames(
            preprocessor.named_transformers_['cat'],
            categorical_features
        )
        inverse_lookup.update(categorical_inverse)

    df_preprocessed = pd.DataFrame(
        result,
        columns=ordinal_features + continuous_features + categorical_names
    )

    if not varianceThreshold:
        return df_preprocessed

    threshold = VarianceThreshold(0.00001)
    threshold.set_output(transform="pandas")

    df_filtered = threshold.fit_transform(df_preprocessed)

    return df_filtered, inverse_lookup


def split(df, column, oneHotX):
    df.dropna(axis=0, inplace=True)
    X, X_inv = preprocess(df.drop(column, axis=1), oneHotEncode=oneHotX)
    y, _ = preprocess(df[[column]], oneHotEncode=False)
    return X, X_inv, y


def groupUnivariateScores(features, scores, inverse):
    grouped = defaultdict(float)
    for feature, score in zip(features, scores):
        grouped[inverse[feature]] += score

    sorted_pairs = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    keys, values = zip(*sorted_pairs)
    return keys, values


def selectFeaturesUnivariate(df, column, oneHotX):
    X, X_inv, y = split(df, column, oneHotX)

    if y.shape[0] == 0 or y.shape[1] == 0:
        return {
            'features': [],
            'scores': [],
            'encoded': {
                'features': [],
                'scores': []
            }
        }

    score_func = f_classif if isCategorical(y[column]) else f_regression

    selector = SelectKBest(score_func=score_func, k='all')

    selector.fit(X, y[column])

    feature_names = X.columns.tolist()
    feature_scores = selector.scores_

    sorted_indices = feature_scores.argsort()[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_feature_scores = [feature_scores[i] for i in sorted_indices]

    grouped_feature_names, grouped_feature_scores = groupUnivariateScores(
        sorted_feature_names, sorted_feature_scores, X_inv
    )

    return {
        'features': grouped_feature_names,
        'scores': grouped_feature_scores,
        'encoded': {
            'features': sorted_feature_names,
            'scores': sorted_feature_scores
        }
    }


def select_features_sequential(df, column, oneHotX):
    X, X_inv, y = split(df, column, oneHotX)

    if y.shape[0] == 0 or y.shape[1] == 0:
        return {
            'fixed': True,
            'features': [],
            'encoded': {
                'features': []
            }
        }

    if X.shape[1] == 1:
        return {
            'fixed': True,
            'features': [X.columns[0]],
            'encoded': {
                'features': [X.columns[0]]
            }
        }

    if isCategorical(y[column]):
        estimator = DecisionTreeClassifier()
    else:
        estimator = DecisionTreeRegressor()

    sfs = SequentialFeatureSelector(estimator=estimator, n_features_to_select='auto', tol=1e-5)
    sfs.fit(X, y)

    encoded_features = sfs.get_feature_names_out().tolist()
    decoded_features = []

    if oneHotX:
        for feature in encoded_features:
            decoded_feature = X_inv[feature]
            if decoded_feature not in decoded_features:
                decoded_features.append(decoded_feature)
    else:
        decoded_features = encoded_features

    return {
        'fixed': True,
        'features': decoded_features,
        'encoded': {
            'features': encoded_features
        }
    }


def select_features_ml(df, column, oneHotX):
    X, X_inv, y = split(df, column, oneHotX)

    if y.shape[0] == 0 or y.shape[1] == 0:
        return {
            'valid': False,
            'features': [],
            'k': 0,
            'kFeatures': [],
            'cumulativeScore': [],
            'cumulativeScoreStd': [],
            'encoded': {
                'features': [],
                'k': 0,
                'kFeatures': [],
                'cumulativeScore': [],
                'cumulativeScoreStd': []
            }
        }

    if X.shape[1] == 1:
        return {
            'valid': False,
            'features': [X.columns[0]],
            'k': 1,
            'kFeatures': [X.columns[0]],
            'cumulativeScore': [0.0],
            'cumulativeScoreStd': [0.0],
            'encoded': {
                'features': [X.columns[0]],
                'k': 1,
                'kFeatures': [X.columns[0]],
                'cumulativeScore': [0.0],
                'cumulativeScoreStd': [0.0]
            }
        }

    if isCategorical(y[column]):
        estimator = DecisionTreeClassifier()
    else:
        estimator = DecisionTreeRegressor()

    rfecv = RFECV(estimator=estimator)
    rfecv.fit(X, y)

    feature_ranks = rfecv.ranking_
    feature_names = X.columns.tolist()

    sorted_features = sorted(zip(feature_names, feature_ranks), key=lambda x: x[1])

    encoded_k = int(rfecv.n_features_)
    encoded_features = [name for name, _ in sorted_features]
    encoded_scores = rfecv.cv_results_["mean_test_score"].tolist()
    encoded_scores_std = rfecv.cv_results_["std_test_score"].tolist()

    decoded_k = 0
    decoded_features = []
    decoded_scores = []
    decoded_scores_std = []

    for i, feature in enumerate(encoded_features):
        decoded_feature = X_inv[feature]
        if decoded_feature in decoded_features:
            if decoded_features[-1] == decoded_feature:
                decoded_scores[-1] = encoded_scores[i]
                decoded_scores_std[-1] = encoded_scores_std[i]
        else:
            decoded_features.append(decoded_feature)
            decoded_scores.append(encoded_scores[i])
            decoded_scores_std.append(encoded_scores_std[i])

        if i == encoded_k - 1:
            decoded_k = len(decoded_features)

    return {
        'valid': True,
        'features': decoded_features,
        'k': decoded_k,
        'kFeatures': decoded_features[:decoded_k],
        'cumulativeScore': decoded_scores,
        'cumulativeScoreStd': decoded_scores_std,
        'encoded': {
            'features': encoded_features,
            'k': encoded_k,
            'kFeatures': encoded_features[:encoded_k],
            'cumulativeScore': encoded_scores,
            'cumulativeScoreStd': encoded_scores_std
        }
    }


def select_features_hybrid(df, column, oneHotX):
    ml = select_features_ml(df, column, oneHotX=oneHotX)
    if not ml['valid']:
        return {
            'fixed': True,
            'features': ml['features']
        }

    univariate = selectFeaturesUnivariate(df, column, oneHotX=oneHotX)

    features = [univariate['features'][0]]

    for ml_feature in ml['kFeatures']:
        if ml_feature not in features:
            features.append(ml_feature)

    best_univ = univariate['scores'][0]
    BEST_THRESH = 0.5

    for univ_feature, univ_score in zip(univariate['features'], univariate['scores']):
        if univ_score < BEST_THRESH * best_univ:
            break

        if univ_feature not in features:
            features.append(univ_feature)

    DESIRED_K = 5
    return {
        'fixed': True,
        'features': features[:DESIRED_K]
    }


def targetType(df, column):
    if column not in df:
        return 'unknown'
    elif isCategorical(df[column]):
        return 'classification'
    else:
        return 'regression'


def oneFeaturesJob(jobNum=0, expDir='exp_base', featuresType='univariate', force=False):
    tu = testUtils.testUtilities()
    tu.registerExpDir(expDir)
    tu.registerFeaturesType(featuresType)
    mc = sdmTools.measuresConfig(tu)
    featuresJobs = mc.getFeaturesJobs()
    if jobNum >= len(featuresJobs):
        print(f"ERROR: jobNum {jobNum} too high")
        quit()
    job = featuresJobs[jobNum]
    featuresFileName = f"{featuresType}.{job['csvFile']}.{job['targetColumn']}.json"
    featuresPath = os.path.join(tu.featuresTypeDir, featuresFileName)
    csvPath = os.path.join(tu.csvLib, job['csvFile'])

    print(f'Processing file \'{job["csvFile"]}\' \'{job["targetColumn"]}\'')

    if not force and os.path.exists(featuresPath):
        print(f"Result {featuresPath} already exists, skipping")
        print("oneFeaturesJob:SUCCESS (skipped)")
        return

    df = loadCsv(csvPath)

    start = time.time()

    data = {
        'type': featuresType,
        'csvFile': job['csvFile'],
        'targetColumn': job['targetColumn'],
        'targetType': targetType(df, job['targetColumn']),
        'origMlScores': job['algInfo']
    }

    one_hot = '1h' in featuresType

    if 'univariate' in featuresType:
        features = selectFeaturesUnivariate(df, job['targetColumn'], one_hot)
        data.update(features)
    elif 'ml' in featuresType:
        if 'seq' in featuresType:
            features = select_features_sequential(df, job['targetColumn'], one_hot)
        elif 'hybrid' in featuresType:
            features = select_features_hybrid(df, job['targetColumn'], one_hot)
        else:
            features = select_features_ml(df, job['targetColumn'], one_hot)

        data.update(features)
    else:
        raise Exception(f'Invalid features type {featuresType}.')

    end = time.time()
    data['elapsedTime'] = end - start

    pp.pprint(data)
    print(f'Saving result to {featuresPath}')

    with open(featuresPath, 'w') as f:
        json.dump(data, f, indent=4)

    print("oneFeaturesJob:SUCCESS")


def main():
    fire.Fire(oneFeaturesJob)


if __name__ == '__main__':
    main()
