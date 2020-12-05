import sys
import warnings
import numpy as np
from pandas import read_csv, concat, DataFrame, get_dummies
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier


np.set_printoptions(threshold=sys.maxsize)
warnings.simplefilter('always')
warnings. filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)


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


def fit_model(X_train, Y_train, X_test, Y_test):
    clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1,
                                            silent=0,
                                            verbosity=1,

                                            learning_rate=0.2,
                                            max_depth=10))

    # You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] to a multilabel
    # format before training.
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)

    clf.fit(X_train, Y_train)

    # y_pred = clf.predict(X_test)

    # print(Y_test)

    return clf

def make_predictions(model, x_test):
    """This function makes predictions using the model on the unseen test dataset
    :param y_test: test labels
    :param model: Sequential model
    :param x_test: unseen test dataset
    :return: predictions in the binary numpy array format
    """

    print(x_test.shape)
    predictions = model.predict_proba(x_test)
    # labels = (np.where(predictions < 0.5, 0, 1)).flatten()
    h1n1_preds = predictions[:, 0].tolist()
    seasonal_preds = predictions[:, 1].tolist()
    print(len(h1n1_preds))
    return model, h1n1_preds, seasonal_preds


def get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds):
    h1n1_score = roc_auc_score(np.array(h1n1_true), np.array(h1n1_preds))
    seasonal_score = roc_auc_score(np.array(seasonal_true), np.array(seasonal_preds))
    average_score = (h1n1_score + seasonal_score) / 2
    return average_score


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

    # df = undersample(df)
    x_train, x_test, y_train, y_test, train_ids, test_ids = split_dataset(df, test_size=0.2, seed=42)

    X_train, Y_train = np.array(x_train), np.array(y_train)
    X_test, Y_test = np.array(x_test), np.array(y_test)

    clf = fit_model(X_train, Y_train, X_test, Y_test)

    model, h1n1_preds, seasonal_preds = make_predictions(clf, X_test)
    h1n1_true, seasonal_true = (Y_test[:, 0]).tolist(), Y_test[:, 1].tolist()
    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(score)

    print('Program execution complete!')
