import lightgbm as lgb
from model import import_data, clean_data, split_dataset, set_df_values, get_scores
from pandas import concat, DataFrame, to_numeric


def submit(test_df, h1n1_clf, seasonal_clf):
    test_df = clean_data(test_df)
    X_test = test_df.iloc[:, 1:]
    test_ids = test_df.iloc[:, 0]

    # X_test, test_ids = X_test.astype(str), test_ids.astype(int)
    X_test = X_test.astype('category')
    h1n1_preds = h1n1_clf.predict(X_test)
    seasonal_preds = seasonal_clf.predict(X_test)

    result_df = concat([test_ids,
                        DataFrame(h1n1_preds, columns=['h1n1_vaccine']),
                        DataFrame(seasonal_preds, columns=['seasonal_vaccine'])],
                       axis=1)
    # print(f'Exporting as pickle...')
    # dump(model, open("classifier.pkl", "wb"))
    result_df.to_csv('Submissions/submission.csv', index=False)
    print('done')

    result_df = concat([test_ids,
                        DataFrame(h1n1_preds, columns=['h1n1_vaccine']),
                        DataFrame(seasonal_preds, columns=['seasonal_vaccine'])],
                       axis=1)
    # print(f'Exporting as pickle...')
    # dump(model, open("classifier.pkl", "wb"))
    result_df.to_csv('Submissions/submission.csv', index=False)
    print('done')


def fit_model(x_train, y_train, x_val):
    y_train['h1n1_vaccine'] = to_numeric(y_train['h1n1_vaccine'])
    y_train['seasonal_vaccine'] = to_numeric(y_train['seasonal_vaccine'])

    h1n1_train = lgb.Dataset(x_train, label=y_train['h1n1_vaccine'])
    seasonal_train = lgb.Dataset(x_train, label=y_train['seasonal_vaccine'])

    # Specifying the parameter
    params = {}
    params['learning_rate'] = 0.03
    params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
    params['objective'] = 'binary'  # Binary target feature
    params['metric'] = 'binary_logloss'  # metric for binary classification
    params['max_depth'] = 100
    # params['max_bin'] = 40
    # params['num_iterations'] = 100
    # train the model
    h1n1_clf = lgb.train(params, h1n1_train, num_boost_round=100)  # train the model on 100 epocs
    h1n1_preds = h1n1_clf.predict(x_val)

    seasonal_clf = lgb.train(params, seasonal_train, num_boost_round=400)
    seasonal_preds = seasonal_clf.predict(x_val)

    return h1n1_preds, seasonal_preds, h1n1_clf, seasonal_clf


if __name__ == '__main__':
    df = import_data(train=True)
    test_df = import_data(features='Datasets/test_set_features.csv', train=False)
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    print(df.dtypes)

    cols = list(df.columns)
    for col in cols[1:-2]:
        df[col] = df[col].astype('category')

    x_train, x_val, y_train, y_val, train_ids, val_ids = split_dataset(df, test_size=0.01, seed=42)

    h1n1_preds, seasonal_preds, h1n1_clf, seasonal_clf = fit_model(x_train, y_train, x_val)

    h1n1_true, seasonal_true = y_val['h1n1_vaccine'].values.tolist(), y_val['seasonal_vaccine'].values.tolist()
    validation_score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Validation Accuracy: {validation_score}')

    submit(test_df, h1n1_clf, seasonal_clf)
    print('done')
