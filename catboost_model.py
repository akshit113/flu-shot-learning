from catboost import CatBoostClassifier
from model import import_data, clean_data, one_hot_encode, split_dataset, set_df_values, make_predictions, get_scores
from sklearn.multiclass import OneVsRestClassifier


def fit_model(x_train, y_train, x_test, y_test):
    ovr = OneVsRestClassifier(estimator=CatBoostClassifier(iterations=10, random_state=1))
    ovr.fit(x_train, y_train)
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
    print('done')
