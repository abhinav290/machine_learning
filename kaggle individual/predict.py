import pandas as pd
import seaborn as sns
import math
import pickle
import numpy as np
from category_encoders import *
from catboost import CatBoostRegressor
# import matplotlib.pyplot as plt
from sklearn import ensemble, preprocessing, model_selection, metrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set()

input_file = "data/training_data.csv"
test_file = "data/test_data.csv"
submission_file = "data/submission.csv"
model_file = "model.sav"


def pre_process(input_file):
    df = pd.read_csv(input_file)
    df.columns = ['instance', 'record_year', 'gender', 'age', 'country', 'city_size', 'profession', 'degree', 'glasses',
                  'hair_color', 'height', 'income']

    columns = ['gender', 'country', 'profession', 'degree']
    for col in columns:
        df[col] = df[col].str.lower()

    ## Showing correlation plot for features
    # print(df.iloc[:, 1:12])
    # correlations = df.iloc[:, 1:12].corr()
    # fig, ax = plt.subplots(figsize=(6,6))
    # sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
    #              square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    # plt.show()

    # Initial Pre-processing
    # Dropping unwanted columns
    df = df.drop(['instance', 'glasses', 'hair_color'], axis=1)
    # df = df.append(df1, ignore_index=True)
    # Processing for column record_year
    # Taking median of the data to replace null values
    mean_year = math.floor(df.record_year.mean())
    df.record_year = df.record_year.fillna(mean_year)

    # Processing for column gender
    df.gender.replace([0, np.nan], '0', inplace=True)
    df.gender = map(lambda x: x.lower(), df.gender)
    df_gender = pd.get_dummies(df.gender)
    df = pd.concat([df, df_gender], axis=1)
    df = df.drop('gender', axis=1)

    # Processing for column age
    # Taking median of the data to replace null values
    median_age = math.floor(df.age.median())
    df.age = df.age.fillna(median_age)

    ## Processing for column country
    # encoder = ce.BinaryEncoder(cols=['country'])
    # country_df = encoder.fit_transform(df.country)
    # df = df.drop('country', axis=1)
    # df = pd.concat([df, country_df], axis=1)


    ## Processing for column profession
    # encoder = ce.BinaryEncoder(cols=['profession'])
    # prof_df = encoder.fit_transform(df.profession)
    # df = df.drop('profession', axis=1)
    # df = pd.concat([df, prof_df], axis=1)

    df.profession = df.profession.str[:5]

    ## Processing for column degree
    df.degree.replace(['no', np.nan], '0', inplace=True)
    # df.degree.replace([np.nan], '0', inplace=True)
    enc = preprocessing.LabelEncoder()
    df.degree = enc.fit_transform(df.degree)

    # Removing predicted outcome variable
    df_income = df.income
    df = df.drop('income', axis=1)
    return df, df_income




## Show graph plots
# plt.figure(figsize=(100, 50))
# plt.xlabel("Country")
# plt.ylabel("Income in euros")
# plt.scatter(df.record_year, df.income, s=1)
# plt.show()

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, params in zip(mean_score, std_score, params):
        print(f'{round(mean, 3)} + or -{round(std, 3)} for the {params}')


def train_model():
    ## Train  model

    df, df_income = pre_process(input_file)
    df_pred_data, df_pred_income = pre_process(test_file)

# Target encoding for Country and Profession
    cat_vars = ['country', 'profession']
    enc = TargetEncoder(cat_vars).fit(df, df_income)
    df = enc.transform(df, df_income)
    df_pred_data = enc.transform(df_pred_data)

# Scaling the data
    scaler = preprocessing.StandardScaler().fit(df)
    df_scaled_data = scaler.transform(df)
    df_pred_scaled_data = scaler.transform(df_pred_data)

# Splitting the data for testing and training
    X_train, X_test, y_train, y_test = model_selection.train_test_split(df_scaled_data, df_income, test_size=0.2,
                                                                        random_state=42)

    parameters = {
        "n_estimators": [5, 10, 50, 100, 250],
        "max_depth": [2, 4, 8, 16, 32, None],
        'min_samples_split': [2, 3]
    }
    # Training and predicting
    #reg = ensemble.RandomForestRegressor(max_depth=32, n_estimators=50, random_state=42)
    reg = CatBoostRegressor(iterations=1400, random_state=42, logging_level='Silent', eval_metric='RMSE', depth=6, bagging_temperature=0.2, learning_rate=0.02)
    reg.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    # cv = model_selection.GridSearchCV(reg, parameters, cv=5, n_jobs=-1, verbose=1)
    # cv.fit(df_scaled_data, df_income.values.ravel())
    # display(cv)

    # Test
    y_pred = reg.predict(df_scaled_data)

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df_income, y_pred)))
    df_pred_income = reg.predict(df_pred_scaled_data)
    return df_pred_income

def store_model(model, model_file):
    pickle.dump(model, open(model_file, 'wb'))


# Store output
def store_output(out_pred):
    df_sub = pd.read_csv(submission_file, index_col=False)
    df_sub['Income'] = out_pred
    with open(submission_file, 'w') as f:
        df_sub.to_csv(f, index=False, line_terminator='\n')


out_pred = train_model()
store_output(out_pred)

