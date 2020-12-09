from pickle import dump

import numpy as np
from catboost import CatBoostClassifier
from model import import_data, clean_data, split_dataset, set_df_values, make_predictions, get_scores
from pandas import concat, DataFrame
from sklearn.multiclass import OneVsRestClassifier


def submit(test_df, model):
    test_df = clean_data(test_df)
    X_test = test_df.iloc[:, 1:]
    test_ids = test_df.iloc[:, 0]
    X_test = np.array(X_test)
    X_test, test_ids = X_test.astype(str), test_ids.astype(int)
    h1n1_preds, seasonal_preds = make_predictions(model, X_test)

    result_df = concat([test_ids,
                        DataFrame(h1n1_preds, columns=['h1n1_vaccine']),
                        DataFrame(seasonal_preds, columns=['seasonal_vaccine'])],
                       axis=1)
    print(f'Exporting as pickle...')
    dump(model, open("classifier.pkl", "wb"))
    result_df.to_csv('Submissions/submission.csv', index=False)
    print('done')


def fit_model(x_train, y_train):
    print('test')
    print(CatBoostClassifier())
    cat_features = ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds', 'behavioral_avoidance',
                    'behavioral_face_mask', 'behavioral_wash_hands', 'behavioral_large_gatherings',
                    'behavioral_outside_home', 'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
                    'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance',
                    'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk', 'opinion_h1n1_sick_from_vacc',
                    'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
                    'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status',
                    'hhs_geo_region', 'census_msa', 'household_adults', 'household_children', 'employment_industry',
                    'employment_occupation']
    ovr = OneVsRestClassifier(estimator=CatBoostClassifier(iterations=230
                                                           , learning_rate=0.15

                                                           , cat_features=cat_features
                                                           , random_state=42))
    ovr.fit(x_train, y_train)
    params = ovr.get_params()
    print('\nHyperparameters:')
    for key, value in params.items():
        print(f'{key[11:]} : {value}')
    return ovr


if __name__ == '__main__':
    df = import_data(train=True)
    test_df = import_data(features='Datasets/test_set_features.csv', train=False)
    cols = list(df.columns)
    set_df_values(df)
    df = clean_data(df)
    x_train, x_val, y_train, y_val, train_ids, val_ids = split_dataset(df, test_size=0.2, seed=42)
    x_train, y_train = x_train.astype(str), y_train.astype(int)
    x_val, y_val = x_val.astype(str), y_val.astype(int)

    model = fit_model(x_train, y_train)
    h1n1_preds, seasonal_preds = make_predictions(model, x_train)
    h1n1_true, seasonal_true = y_train['h1n1_vaccine'].values.tolist(), y_train['seasonal_vaccine'].values.tolist()
    train_score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Training Accuracy: {train_score}')

    h1n1_preds, seasonal_preds = make_predictions(model, x_val)
    h1n1_true, seasonal_true = y_val['h1n1_vaccine'].values.tolist(), y_val['seasonal_vaccine'].values.tolist()
    validation_score = get_scores(h1n1_true, h1n1_preds, seasonal_true, seasonal_preds)
    print(f'Validation Accuracy: {validation_score}')

    """
    0.8515016459915379 - epochs = 230, learning_rate=0.1
    """
    submit(test_df, model)
    print('done')
