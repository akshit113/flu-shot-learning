import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from model import import_data, clean_data, split_dataset, set_df_values, one_hot_encode
from sklearn.metrics import roc_auc_score


def get_model(input_size, output_size, magic='tanh'):
    """This function creates a baseline feedforward neural network with of given input size and output size
        using magic activation function.
    :param input_size: number of columns in x_train
    :param output_size: no of columns in one hpt
    :param magic: activation function
    :return:Sequential model
    """
    mlmodel = Sequential()
    mlmodel.add(Dense(input_size, input_dim=input_size, activation=magic))
    # kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(1e-4),
    # activity_regularizer=l1(1e-5)))
    # mlmodel.add(LeakyReLU(alpha=0.1))
    mlmodel.add(Dense(64, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(254, activation=magic))
    mlmodel.add(Dense(324, activation=magic))
    mlmodel.add(Dense(512, activation=magic))

    mlmodel.add(Dense(output_size, activation='sigmoid'))

    # Setting optimizer
    # mlmodel.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    opt = SGD(lr=0.001)
    mlmodel.compile(loss="binary_crossentropy", optimizer='adam', metrics=['binary_accuracy'])
    return mlmodel


def fit_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    """fits the model created in the get_model function on x_train, y_train and evaluates the model performance on
    x_test and y_test using the batch size and epochs paramters
    :param model: Sequential model
    :param x_train: training data
    :param y_train: training label
    :param x_test: testing data
    :param y_test: testing label
    :param batch_size: amount of training data (x_train) fed to the model
    :param epochs: number of times the entire dataset is passed through the network
    :return: tuple of validation_accuracy and validation_loss
    """
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_acc)
    print('Test Loss:', test_loss)
    return test_acc, test_loss


def make_predictions(model, x_test):
    """This function makes predictions using the model on the unseen test dataset
    :param y_test: test labels
    :param model: Sequential model
    :param x_test: unseen test dataset
    :return: predictions in the binary numpy array format
    """
    predictions = model.predict(x_test)
    # labels = (np.where(predictions < 0.5, 0, 1)).flatten()
    if isinstance(predictions, list):
        h1n1_preds = predictions[0]
        seasonal_preds = predictions[1]

    else:
        h1n1_preds = predictions[:, 0].tolist()
        seasonal_preds = predictions[:, 1].tolist()

    return h1n1_preds, seasonal_preds


def get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds):
    h1n1_score = roc_auc_score(np.array(h1n1_true), np.array(h1n1_preds))
    seasonal_score = roc_auc_score(np.array(seasonal_true), np.array(seasonal_preds))
    average_score = (h1n1_score + seasonal_score) / 2
    return average_score


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
    # X_train, Y_train = np.array(x_train), np.array(y_train)
    # X_val, Y_val = np.array(x_val), np.array(y_val)

    model = get_model(input_size=118, output_size=2)
    x_train = np.asarray(x_train).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    x_val = np.asarray(x_val).astype(np.float32)
    y_val = np.asarray(y_val).astype(np.float32)

    test_acc, test_loss = fit_and_evaluate(model, x_train, y_train, x_val, y_val, batch_size=64, epochs=1)

    h1n1_preds, seasonal_preds = make_predictions(model, x_train)
    h1n1_true, seasonal_true = y_train[:, 0].tolist(), y_train[:, 1].tolist()
    train_score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Training Accuracy: {train_score}')

    h1n1_preds, seasonal_preds = make_predictions(model, x_val)
    h1n1_true, seasonal_true = y_val[:, 0].tolist(), y_val[:, 1].tolist()
    validation_score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Validation Accuracy: {validation_score}')

    print('program execution complete')
