import numpy as np
from model import import_data, clean_data, split_dataset, set_df_values, one_hot_encode, make_predictions, get_scores
from pandas import to_numeric
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

if __name__ == '__main__':
    df = import_data(train=True)
    test_df = import_data(features='Datasets/test_set_features.csv', train=False)
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    print(df.dtypes)
    ohe_cols = cols[1:36]
    # print(ohe_cols)
    df = one_hot_encode(df, colnames=ohe_cols)

    x_train, x_val, y_train, y_val, train_ids, val_ids = split_dataset(df, test_size=0.3, seed=42)
    X_train, Y_train = np.array(x_train), np.array(y_train)
    X_val, Y_val = np.array(x_val), np.array(y_val)

    clf = OneVsRestClassifier(LogisticRegression(verbose=3))  # 84%
    y_train['h1n1_vaccine'] = to_numeric(y_train['h1n1_vaccine'])
    y_train['seasonal_vaccine'] = to_numeric(y_train['seasonal_vaccine'])

    clf.fit(x_train, y_train)

    h1n1_preds, seasonal_preds = make_predictions(clf, X_val)
    h1n1_true, seasonal_true = (Y_val[:, 0]).tolist(), Y_val[:, 1].tolist()

    score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print('Validation Accuracy = ', score)
