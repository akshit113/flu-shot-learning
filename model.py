import sys
import warnings

import numpy as np
from pandas import read_csv, concat, DataFrame, get_dummies
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

np.set_printoptions(threshold=sys.maxsize)
warnings.simplefilter('always')
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)


def import_data(features='Datasets/training_set_features.csv',
                labels='Datasets/training_set_labels.csv', train=True):
    """Import dataset and remove row numbers column
    :return: dataframe
    """

    df = read_csv(features)
    if train:
        df_labels = read_csv(labels)
        df = concat([df, df_labels.iloc[:, [1, 2]]], axis=1)
    # cols = list(df.columns)
    # print("\n".join(cols))
    return df


def set_df_values(df):
    cols = list(df.columns)
    # ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'household_adults', 'household_children', 'employment_industry', 'employment_occupation', 'h1n1_vaccine', 'seasonal_vaccine']
    for col in cols:
        ls = set(df[col].values.tolist())
        vals = {x for x in ls if x == x}
        # print(f'{col}:{vals}')
    # cols=['h1n1_concern','h1n1_knowledge','behavioral_avoidance','behavioral_face_mask','behavioral_wash_hands',
    # 'behavioral_large_gatherings','behavioral_outside_home','behavioral_touch_face','doctor_recc_h1n1',


def clean_data(df):
    """Clean dataframe and impute missing values by mode
    :param df: input dataframe
    :return: clean dataframe
    """
    cols = list(df.columns)
    mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df = (DataFrame(mode_imp.fit_transform(df)))
    df.columns = cols
    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """

    for col in colnames:
        # print(col)
        oh_df = get_dummies(df[col], prefix=col, drop_first=True)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    missing = (df.isnull().values.any())
    while missing:
        df = df.dropna()
        print(df.isnull().sum())
        missing = (df.isnull().values.any())

    resp_df = df['respondent_id'].copy()
    df.drop(['respondent_id'], axis=1, inplace=True)
    df = concat([resp_df, df], axis=1)
    # print(list(df.columns))
    # print(df.shape)
    return df


def split_dataset(df, test_size, seed):
    """This function randomly splits (using seed) train data into training set and validation set. The test size
    paramter specifies the ratio of input that must be allocated to the test set
    :param df: one-hot encoded dataset
    :param test_size: ratio of test-train data
    :param seed: random split
    :return: training and validation data
    """
    ncols = np.size(df, 1)
    X = df.iloc[:, range(0, ncols - 2)]

    Y = df.iloc[:, ncols - 2:]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    train_ids = x_train['respondent_id'].copy()
    test_ids = x_test['respondent_id'].copy()
    x_train.drop(['respondent_id'], axis=1, inplace=True)
    x_test.drop(['respondent_id'], axis=1, inplace=True)
    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    return x_train, x_test, y_train, y_test, train_ids, test_ids


def fit_model(X_train, Y_train):
    clf = OneVsRestClassifier(XGBClassifier())
    Y_train = Y_train.astype('int')
    clf.fit(X_train, Y_train)
    print('')
    return clf


def make_predictions(model, x_test):
    """This function makes predictions using the model on the unseen test dataset
    :param y_test: test labels
    :param model: Sequential model
    :param x_test: unseen test dataset
    :return: predictions in the binary numpy array format
    """
    predictions = model.predict_proba(x_test)
    # labels = (np.where(predictions < 0.5, 0, 1)).flatten()
    h1n1_preds = predictions[:, 0].tolist()
    seasonal_preds = predictions[:, 1].tolist()

    return model, h1n1_preds, seasonal_preds


def get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds):
    h1n1_score = roc_auc_score(np.array(h1n1_true), np.array(h1n1_preds))
    seasonal_score = roc_auc_score(np.array(seasonal_true), np.array(seasonal_preds))
    average_score = (h1n1_score + seasonal_score) / 2
    return average_score


def submit(test_df):
    X_test = test_df.iloc[:, 1:]
    test_ids = test_df.iloc[:, 0]
    X_test = np.array(X_test)

    model, h1n1_preds, seasonal_preds = make_predictions(clf, X_test)
    result_df = concat([test_ids,
                        DataFrame(h1n1_preds, columns=['h1n1_vaccine']),
                        DataFrame(seasonal_preds, columns=['seasonal_vaccine'])],
                       axis=1)
    result_df.to_csv('Submissions/submission.csv', index=False)


if __name__ == '__main__':
    df = import_data(train=True)
    test_df = import_data(features='Datasets/test_set_features.csv', train=False)
    # print(list(df.columns))
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    test_df = clean_data(test_df)
    ohe_cols = cols[1:36]
    # print(ohe_cols)
    df = one_hot_encode(df, colnames=ohe_cols)
    test_df = one_hot_encode(test_df, colnames=ohe_cols)
    x_train, x_val, y_train, y_val, train_ids, val_ids = split_dataset(df, test_size=0.3, seed=42)
    X_train, Y_train = np.array(x_train), np.array(y_train)
    X_val, Y_val = np.array(x_val), np.array(y_val)

    clf = fit_model(X_train, Y_train)

    predictions = clf.predict_proba(X_train)
    h1n1_preds = predictions[:, 0].tolist()
    seasonal_preds = predictions[:, 1].tolist()
    # print(len(h1n1_preds))
    h1n1_true, seasonal_true = (Y_train[:, 0]).tolist(), Y_train[:, 1].tolist()
    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print('Training Accuracy = ', score)

    model, h1n1_preds, seasonal_preds = make_predictions(clf, X_val)
    h1n1_true, seasonal_true = (Y_val[:, 0]).tolist(), Y_val[:, 1].tolist()
    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print('Validation Accuracy = ', score)

    submit(test_df)

    print('Program execution complete!')
