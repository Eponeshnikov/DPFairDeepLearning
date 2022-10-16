import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessing_adult(path, use_age=False, age_val=(10, 65), test_size=0.3, seed=0):
    def normalize(df_):
        df_.fnlwgt = (df_.fnlwgt - np.mean(df_.fnlwgt)) / np.std(df_.fnlwgt)

        df_['capital-gain'] = (df_['capital-gain'] -
                               np.mean(df_['capital-gain'])) / np.std(df_['capital-gain'])
        df_['capital-loss'] = (df_['capital-loss'] -
                               np.mean(df_['capital-loss'])) / np.std(df_['capital-loss'])

        # apply min max scaler
        df_['hours-per-week'] = (df_['hours-per-week'] - np.min(df_['hours-per-week'])) / (
                np.max(df_['hours-per-week']) - np.min(df_['hours-per-week']))

        df_['education-num'] = (df_['education-num'] - np.min(df_['education-num'])) / \
                               (np.max(df_['education-num']) - np.min(df_['education-num']))
        return df_

    def x_y_s(df_):
        if use_age:
            S = (df_["age"].values < age_val[1]) & (df_["age"].values > age_val[0])
        else:
            S = df_['gender_Male'].values

        # delete old columns
        del df_['workclass']
        del df_['education']
        del df_['marital-status']
        del df_['occupation']
        del df_['relationship']
        del df_['race']
        del df_['sex']
        del df_['native-country']

        del df_['income']

        # df_.to_csv('preprocessing/adult.csv')
        del df_['gender_Male']

        X = df_.drop('income_>50K', axis=1).values[0:n_data, :]
        y = df_['income_>50K'].values[0:n_data]
        S = S[0:n_data]
        return np.array(X), np.array(y), np.array(S)

    headers = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex," \
              "capital-gain,capital-loss,hours-per-week,native-country,income".split(",")
    df_train = pd.read_csv(f'{path}/adult.data', names=headers, sep=r'\s*,\s*',
                           engine='python', na_values="?")
    df_test = pd.read_csv(f'{path}/adult.test', names=headers, sep=r'\s*,\s*', skiprows=[0],
                          engine='python', na_values="?")
    df_test['income'] = df_test.income.str.rstrip('.')
    df = pd.concat([df_train, df_test], ignore_index=True)
    del df_test
    del df_train
    n_data = len(df)

    df = df.merge(pd.get_dummies(df.sex, drop_first=True,
                                 prefix='gender'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['native-country'], drop_first=True,
                                 prefix='native_country'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['marital-status'], drop_first=True,
                                 prefix='marital_status'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(
        df['race'], drop_first=True, prefix='race'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['education'], drop_first=True,
                                 prefix='education'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['occupation'], drop_first=True,
                                 prefix='occupation'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['relationship'], drop_first=True,
                                 prefix='relationship'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['workclass'], drop_first=True,
                                 prefix='workclass'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.income, drop_first=True,
                                 prefix='income'), left_index=True, right_index=True)

    # feature scaling
    # apply stander scaler

    #df_train, df_test = df.head(int((1 - test_size) * n_data)), df.tail(int(test_size * n_data))
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    del df
    df_train = normalize(df_train)
    df_test = normalize(df_test)
    X_train, y_train, S_train = x_y_s(df_train)
    X_test, y_test, S_test = x_y_s(df_test)
    return X_train, y_train, S_train, X_test, y_test, S_test


def generate_features_values(prefix, size, index=1):
    a = np.arange(index, size + index)
    return [prefix + str(i) for i in a]
