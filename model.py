import warnings

import numpy as np
from pandas import read_csv, cut, concat, DataFrame
from sklearn.impute import SimpleImputer


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
    print(' ')
    print('')
    return df


def set_df_values(df):
    cols = list(df.columns)
    #['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal', 'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region', 'census_msa', 'household_adults', 'household_children', 'employment_industry', 'employment_occupation', 'h1n1_vaccine', 'seasonal_vaccine']
    for col in cols:
        ls = set(df[col].values.tolist())
        vals = {x for x in ls if x == x}
        print(f'{col}:{vals}')
    #cols=['h1n1_concern','h1n1_knowledge','behavioral_avoidance','behavioral_face_mask','behavioral_wash_hands',
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



if __name__ == '__main__':
    df = import_data(features='Datasets/training_set_features.csv',
                     labels='Datasets/training_set_labels.csv')
    print(list(df.columns))

    set_df_values(df)
    df = clean_data(df)
    test_df = clean_data(df)
    print(df)
    # df = normalize_columns(df, colnames=['age', 'MonthlyIncome', 'NumberOfDependents'])
    # df = one_hot_encode(df, test_df, colnames=['ages'])
    # df = undersample(df)
    # X, Y, x_train, x_test, y_train, y_test = split_dataset(df, test_size=0.2, seed=42)
    # missing = (X.isnull().values.any())
    # if missing:
    #     print(X.isnull().sum())
    # X_train, Y_train = np.array(x_train), np.array(y_train)
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
