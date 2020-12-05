import sys
import warnings

import numpy as np
from pandas import read_csv, concat, DataFrame, get_dummies
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


def import_data(features, labels, train=True):
    """Import dataset and remove row numbers column
    :return: dataframe
    """
    df = read_csv(features)
    df_labels = read_csv(labels)
    df = concat([df, df_labels.iloc[:, [1, 2]]], axis=1)

    cols = list(df.columns)
    print("\n".join(cols))
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
    print('done')


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
    print(list(df.columns))
    print(df.shape)
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
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test, train_ids, test_ids


if __name__ == '__main__':
    df = import_data(features='Datasets/training_set_features.csv',
                     labels='Datasets/training_set_labels.csv')
    print(list(df.columns))
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    test_df = clean_data(df)
    ohe_cols = cols[1:36]
    print(ohe_cols)
    df = one_hot_encode(df, colnames=ohe_cols)
    print(df)

    # df = undersample(df)
    x_train, x_test, y_train, y_test, train_ids, test_ids = split_dataset(df, test_size=0.2, seed=42)

    X_train, Y_train = np.array(x_train), np.array(y_train)
    # classifier = get_model(X.shape[1], 1, magic='sigmoid')
    # start = datetime.now()
    # batch = 1024
    # epochs = 5  # 100
    #
    # test_acc, test_loss = fit_and_evaluate(classifier, X_train, Y_train, x_test, y_test, batch_size=batch,
    #                                        epochs=epochs)
    # end = datetime.now()
    # total_seconds = (end - start).seconds
    # minute, seconds = divmod(total_seconds, 60)
    # print(f'\nTraining Time: {minute}:{seconds}')
    # avg = total_seconds / epochs
    # print(f'\nAverage time per epoch: {avg}')
    #
    # print(test_acc, test_loss)
    #
    # y_hat, model = make_predictions(classifier, x_test)
    #
    # roc_auc_score = get_metrics(y_test, y_hat, 'roc_auc_score', 'macro')
    # print(roc_auc_score)
    #
    # fname = 'nn_logs.xlsx'
    # export_flag = write_logs(fname, roc_auc_score)
    # export(model)
    #
    # if export_flag:
    #     export(model)
    #
    # jcard_score = get_metrics(y_test, y_hat, 'jaccard_score', 'binary')
    # f1 = get_metrics(y_test, y_hat, 'f1_score', 'binary')
    # precision = get_metrics(y_test, y_hat, 'precision_score', 'binary')
    # recall = get_metrics(y_test, y_hat, 'recall_score', 'binary')

    print('Program execution complete!')
