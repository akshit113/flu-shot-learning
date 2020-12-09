from pickle import dump

import numpy as np
from catboost import CatBoostClassifier
from model import import_data, clean_data, one_hot_encode, split_dataset, set_df_values, make_predictions, get_scores
from pandas import concat, DataFrame
from sklearn.multiclass import OneVsRestClassifier


def submit(test_df, model):
    test_df = clean_data(test_df)
    ohe_cols = cols[1:36]
    test_df = one_hot_encode(test_df, colnames=ohe_cols)
    X_test = test_df.iloc[:, 1:]
    test_ids = test_df.iloc[:, 0]
    X_test = np.array(X_test)

    h1n1_preds, seasonal_preds = make_predictions(model, X_test)

    result_df = concat([test_ids,
                        DataFrame(h1n1_preds, columns=['h1n1_vaccine']),
                        DataFrame(seasonal_preds, columns=['seasonal_vaccine'])],
                       axis=1)
    print(f'Exporting as pickle...')
    dump(model, open("classifier.pkl", "wb"))
    result_df.to_csv('Submissions/submission.csv', index=False)
    print('done')


def fit_model(x_train, y_train, x_test, y_test):
    print('test')
    print(CatBoostClassifier())
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    ovr = OneVsRestClassifier(estimator=CatBoostClassifier(iterations=100
                                                           , learning_rate=0.1
                                                           , random_state=42))
    ovr.fit(x_train, y_train)
    print('done')
    return ovr


if __name__ == '__main__':
    df = import_data(train=True)
    test_df = import_data(features='Datasets/test_set_features.csv', train=False)
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    ohe_cols = cols[1:36]
    # print(ohe_cols)
    df = one_hot_encode(df, colnames=ohe_cols)

    x_train, x_val, y_train, y_val, train_ids, val_ids = split_dataset(df, test_size=0.3, seed=42)

    model = fit_model(x_train, y_train, x_val, y_val)

    h1n1_preds, seasonal_preds = make_predictions(model, x_train)
    h1n1_true, seasonal_true = y_train['h1n1_vaccine'].values.tolist(), y_train['seasonal_vaccine'].values.tolist()
    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Training Accuracy: {score}')

    h1n1_preds, seasonal_preds = make_predictions(model, x_val)
    h1n1_true, seasonal_true = y_val['h1n1_vaccine'].values.tolist(), y_val['seasonal_vaccine'].values.tolist()
    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Validation Accuracy: {score}')

    submit(test_df, model)
    print('done')
