import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


def preprocessing_adult(path, predattr='income_>50K', use_age=False, age_val=(10, 65), test_size=0.3, seed=0):
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

        X = df_.drop(predattr, axis=1).values[0:n_data, :]
        y = df_[predattr].values[0:n_data]
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

    # df_train, df_test = df.head(int((1 - test_size) * n_data)), df.tail(int(test_size * n_data))
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)
    del df
    df_train = normalize(df_train)
    df_test = normalize(df_test)
    X_train, y_train, S_train = x_y_s(df_train)
    X_test, y_test, S_test = x_y_s(df_test)
    return X_train, y_train, S_train, X_test, y_test, S_test


def preprocessing_german(path, predattr='Risk_good', test_size=0.3, seed=1):
    """ Convert data to numeric values """
    header = 'Checkin account,Duration,Credit history,Purpose,Credit amount,Savings,Present employment,' \
             'Installment rate,Sex,Other debtors,Residence duration,Property,Age,Other installment,Housing,' \
             'Credits number,Job,Number of people,Telephone,Foreign worker,Risk'.split(',')
    data = pd.read_csv(f'{path}/german.data', names=header, delimiter=' ')
    purposes = ['car', 'car_new', 'furniture/equipment', 'radio/TV', 'domestic appliances',
                'repairs', 'education', 'vacation', 'retraining', 'business', 'others']

    data["Sex"] = data["Sex"].replace(["A91", "A92", "A93", "A94", "A95"], [
        "male", "female", "male", "male", "female"])
    data["Risk"] = data["Risk"].replace([1, 2], ["good", "bad"])
    data["Checkin account"] = data["Checkin account"].replace(
        ["A11", "A12", "A13", "A14"], ['little', 'moderate', 'hight', None])
    data["Credit history"] = data["Credit history"].replace(
        ["A30", "A31", "A32", "A33", "A34"], np.arange(5))
    # data["Purpose"] = data["Purpose"].replace(generate_features_values('A4', 11, 0), np.arange(11))
    data["Purpose"] = data["Purpose"].replace(
        generate_features_values('A4', 11, 0), purposes)
    # data["Savings"] = data["Savings"].replace(generate_features_values('A6', 6), np.arange(6))
    data["Savings"] = data["Savings"].replace(generate_features_values(
        'A6', 5), ['little', 'moderate', 'rich', 'quite_rich', None])
    data["Present employment"] = data["Present employment"].replace(
        generate_features_values('A7', 6), np.arange(6))
    # data["Other debtors"] = data["Other debtors"].replace(generate_features_values('A10', 4), np.arange(4))
    data["Other debtors"] = data["Other debtors"].replace(
        generate_features_values('A10', 3), [None, 'co_appli', 'guarantor'])
    data["Property"] = data["Property"].replace(
        generate_features_values('A12', 5), np.arange(5))
    # data["Other installment"] = data["Other Installement"].replace(generate_features_values('A14', 4), np.arange(4))
    data["Other installment"] = data["Other installment"].replace(
        generate_features_values('A14', 3), ['bank', 'stores', 'none'])
    # data["Housing"] = data["Housing"].replace(generate_features_values('A15', 3), np.arange(3))
    data["Housing"] = data["Housing"].replace(
        generate_features_values('A15', 3), ['rent', 'own', 'free'])
    data["Job"] = data["Job"].replace(
        generate_features_values('A17', 5), np.arange(5))
    data["Telephone"] = data["Telephone"].replace(
        generate_features_values('A19', 2), [None, "yes"])
    data["Foreign worker"] = data["Foreign worker"].replace(
        generate_features_values('A20', 2), ["yes", None])
    data["Credit amount"] = (data["Credit amount"] - np.mean(data["Credit amount"])) / np.std(
        data["Credit amount"])  # np.log(data["Credit amount"])

    data["Duration"] = (data["Duration"] - np.mean(data["Duration"])) / np.std(
        data["Duration"])  # np.log(data["Credit amount"])

    data["Age"] = (data["Age"] - np.mean(data["Age"])) / np.std(data["Age"])  # np.log(data["Credit amount"])

    # Purpose of Dummies
    data = data.merge(pd.get_dummies(data.Purpose, drop_first=True,
                                     prefix='Purpose'), left_index=True, right_index=True)

    # Dummies of Sex
    data = data.merge(pd.get_dummies(data.Sex, drop_first=True,
                                     prefix='Sex'), left_index=True, right_index=True)

    # Dummies of Other debtors
    data = data.merge(pd.get_dummies(
        data["Other debtors"], drop_first=True, prefix='Other_debtors'), left_index=True, right_index=True)

    # Dummies of Other installment
    data = data.merge(pd.get_dummies(
        data["Other installment"], drop_first=True, prefix='Other_install'), left_index=True, right_index=True)

    # Dummies of Foreign worker
    data = data.merge(pd.get_dummies(
        data["Foreign worker"], drop_first=True, prefix='Foreign worker'), left_index=True, right_index=True)

    # Dummies of Housing
    data = data.merge(pd.get_dummies(
        data["Housing"], drop_first=True, prefix='Housing'), left_index=True, right_index=True)

    # Dummies of Telephone
    data = data.merge(pd.get_dummies(
        data["Telephone"], drop_first=True, prefix='Telephone'), left_index=True, right_index=True)

    # Dummies of Telephone
    data = data.merge(pd.get_dummies(
        data["Savings"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)

    # Dummies of Checkin account
    data = data.merge(pd.get_dummies(
        data["Checkin account"], drop_first=True, prefix='Checkin account'), left_index=True, right_index=True)

    # Dummies of Checkin account
    data = data.merge(pd.get_dummies(
        data["Risk"], drop_first=True, prefix='Risk'), left_index=True, right_index=True)

    # Delete old features
    del data["Other installment"]
    del data["Other debtors"]
    del data["Sex"]
    del data["Purpose"]
    del data["Foreign worker"]
    del data["Housing"]
    del data["Telephone"]
    del data["Savings"]
    del data["Checkin account"]
    del data["Risk"]

    data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed)
    S_train = np.array(data_train["Sex_male"].values)
    S_test = np.array(data_test["Sex_male"].values)
    X_train = np.array(data_train.drop(predattr, 1).values)
    X_test = np.array(data_test.drop(predattr, 1).values)
    y_train = np.array(data_train[predattr].values)
    y_test = np.array(data_test[predattr].values)

    return X_train, y_train, S_train, X_test, y_test, S_test


def preprocessing_celeba(path, predattr, sensattr, test_size=0.3, seed=2):
    annotations_path = f'{path}/combined_annotation.txt'
    captions_path = f'{path}/captions.json'

    annotations_df = pd.read_csv(annotations_path, delimiter=' ')
    captions_dict = json.load(open(captions_path))
    captions_series = pd.Series({key: value['overall_caption'] for key, value in captions_dict.items()},
                                name='captions')
    merged_df = pd.merge(annotations_df, captions_series.to_frame(), left_on='img_name', right_index=True, how='left')

    merged_df['Eyeglasses'] = np.where(merged_df['Eyeglasses'] != 0, 1, 0)
    merged_df['Smiling'] = np.where(merged_df['Smiling'] != 0, 1, 0)
    merged_df['No_Beard'] = np.where(merged_df['No_Beard'] != 0, 1, 0)
    merged_df['Bangs'] = merged_df['Bangs'].apply(lambda x: 1 if x > 2 else 0)
    merged_df['Young'] = merged_df['Young'].apply(lambda x: 1 if x > 2 else 0)
    merged_df = merged_df.sample(frac=0.5, random_state=seed).reset_index()

    model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
    emb = model.encode(merged_df['captions'])
    del model
    merged_df['captions_vec'] = [i for i in emb]
    train, test = train_test_split(merged_df, test_size=test_size, random_state=seed)
    X_train, y_train, S_train = np.stack(train['captions_vec'].to_list(), axis=0), train[predattr].to_numpy(), \
        train[sensattr].to_numpy()
    X_test, y_test, S_test = np.stack(test['captions_vec'].to_list(), axis=0), test[predattr].to_numpy(), \
        test[sensattr].to_numpy()
    return X_train, y_train, S_train, X_test, y_test, S_test


def generate_features_values(prefix, size, index=1):
    a = np.arange(index, size + index)
    return [prefix + str(i) for i in a]
