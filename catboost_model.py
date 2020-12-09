import numpy as np
from model import import_data, clean_data, one_hot_encode, split_dataset, set_df_values


def fit_model(X_train,Y_train):
    














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
    X_train, Y_train = np.array(x_train), np.array(y_train)
    X_val, Y_val = np.array(x_val), np.array(y_val)
