import pandas as pd1
import numpy as np
import pandas as pd


def preprocessing_adult(df, n_data, use_age=False, include_s = True):

    df = df.replace([" ?"], [None])

    df['Education'].replace(' Preschool', 'Preschool', inplace=True)
    df['Education'].replace(' 10th', '10th', inplace=True)
    df['Education'].replace(' 11th', '11th', inplace=True)
    df['Education'].replace(' 12th', '12th', inplace=True)
    df['Education'].replace(' 1st-4th', '1st_4th', inplace=True)
    df['Education'].replace(' 5th-6th', '5th_6th', inplace=True)
    df['Education'].replace(' 7th-8th', '7th_8th', inplace=True)
    df['Education'].replace(' 9th', '9th', inplace=True)
    df['Education'].replace(' HS-grad', 'HighGrad', inplace=True)
    df['Education'].replace(' Some-college', 'SomeCollege', inplace=True)
    df['Education'].replace(' Assoc-acdm', 'AssocAcdm', inplace=True)
    df['Education'].replace(' Assoc-voc', 'AssocVoc', inplace=True)
    df['Education'].replace(' Bachelors', 'Bachelors', inplace=True)
    df['Education'].replace(' Masters', 'Masters', inplace=True)
    df['Education'].replace(' Prof-school', 'ProfSchool', inplace=True)
    df['Education'].replace(' Doctorate', 'Doctorate', inplace=True)

    df['marital-status'].replace(' Never-married',
                                 'NeverMarried', inplace=True)
    df['marital-status'].replace([' Married-AF-spouse'],
                                 'MarriedAFspous', inplace=True)
    df['marital-status'].replace([' Married-civ-spouse'],
                                 'MarriedCivSpouse', inplace=True)
    df['marital-status'].replace([' Married-spouse-absent'],
                                 'MarriedSpouseAbsent', inplace=True)
    df['marital-status'].replace([' Separated'], 'Separated', inplace=True)
    df['marital-status'].replace([' Divorced'], 'Divorced', inplace=True)
    df['marital-status'].replace([' Widowed'], 'Widowed', inplace=True)

    df = df.merge(pd.get_dummies(df.Sex, drop_first=True,
                  prefix='gender'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['Native-country'], drop_first=True,
                  prefix='native_country'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['marital-status'], drop_first=True,
                  prefix='marital_status'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(
        df['Race'], drop_first=True, prefix='race'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['Education'], drop_first=True,
                  prefix='education'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['Occupation'], drop_first=True,
                  prefix='occupation'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['Relationship'], drop_first=True,
                  prefix='relationship'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df['aworkclass'], drop_first=True,
                  prefix='aworkclass'), left_index=True, right_index=True)
    #df = df.merge(pd.get_dummies(df['Education-num'], drop_first=True, prefix='Education-num'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.Outcome, drop_first=True,
                  prefix='outcome'), left_index=True, right_index=True)

    # feature scaling
    # apply stander scaler
    #df.Age = (df.Age - np.mean(df.Age))/np.std(df.Age)

    df.fnlwgt = (df.fnlwgt - np.mean(df.fnlwgt))/np.std(df.fnlwgt)

    df['Capital-gain'] = (df['Capital-gain'] -
                          np.mean(df['Capital-gain']))/np.std(df['Capital-gain'])
    df['Capital-loss'] = (df['Capital-loss'] -
                          np.mean(df['Capital-loss']))/np.std(df['Capital-loss'])

    # apply min max scaler
    df['hours-per-week'] = (df['hours-per-week'] - np.min(df['hours-per-week']))/(
        np.max(df['hours-per-week']) - np.min(df['hours-per-week']))

    df['Education-num'] = (df['Education-num'] - np.min(df['Education-num'])) / \
        (np.max(df['Education-num']) - np.min(df['Education-num']))

    if use_age:
        S = df["Age"].values > 65
    else:
        S = df['gender_ Male'].values

    # delete old columns
    del df['aworkclass']
    del df['Education']
    del df['marital-status']
    del df['Occupation']
    del df['Relationship']
    del df['Race']
    del df['Sex']
    del df['Native-country']
 
    del df['Outcome']

    df.to_csv('preprocessing/adult.csv')
    del df['gender_ Male']

    X = df.drop('outcome_ >50K', axis=1).values[0:n_data, :]
    y = df['outcome_ >50K'].values[0:n_data]
    S = S[0:n_data]
    return np.array(X), np.array(y), np.array(S), df


def preprocessing_german(data, include_s = True):
    """ Convert data to numeric values """
    purposes = ['car', 'car_new', 'furniture/equipment', 'radio/TV', 'domestic appliances',
                'repairs', 'education', 'vacation', 'retraining', 'business', 'others']

    data["Sex"] = data["Sex"].replace(["A91", "A92", "A93", "A94", "A95"], [
                                      "male", "female", "male", "male", "female"])
    data["Risk"] = data["Risk"].replace([1, 2], ["good", "bad"])
    data["Checkin account"] = data["Checkin account"].replace(
        ["A11", "A12", "A13", "A14"], ['little', 'moderate', 'hight', None])
    data["Credit history"] = data["Credit history"].replace(
        ["A30", "A31", "A32", "A33", "A34"], np.arange(5))
    #data["Purpose"] = data["Purpose"].replace(generate_features_values('A4', 11, 0), np.arange(11))
    data["Purpose"] = data["Purpose"].replace(
        generate_features_values('A4', 11, 0), purposes)
    #data["Savings"] = data["Savings"].replace(generate_features_values('A6', 6), np.arange(6))
    data["Savings"] = data["Savings"].replace(generate_features_values(
        'A6', 5), ['little', 'moderate', 'rich', 'quite_rich', None])
    data["Present employment"] = data["Present employment"].replace(
        generate_features_values('A7', 6), np.arange(6))
    #data["Other debtors"] = data["Other debtors"].replace(generate_features_values('A10', 4), np.arange(4))
    data["Other debtors"] = data["Other debtors"].replace(
        generate_features_values('A10', 3), [None, 'co_appli', 'guarantor'])
    data["Property"] = data["Property"].replace(
        generate_features_values('A12', 5), np.arange(5))
    #data["Other installment"] = data["Other Installement"].replace(generate_features_values('A14', 4), np.arange(4))
    data["Other installment"] = data["Other installment"].replace(
        generate_features_values('A14', 3), ['bank', 'stores', 'none'])
    #data["Housing"] = data["Housing"].replace(generate_features_values('A15', 3), np.arange(3))
    data["Housing"] = data["Housing"].replace(
        generate_features_values('A15', 3), ['rent', 'own', 'free'])
    data["Job"] = data["Job"].replace(
        generate_features_values('A17', 5), np.arange(5))
    data["Telephone"] = data["Telephone"].replace(
        generate_features_values('A19', 2), [None, "yes"])
    data["Foreign worker"] = data["Foreign worker"].replace(
        generate_features_values('A20', 2), ["yes", None])
    data["Credit amount"] = (data["Credit amount"] - np.mean(data["Credit amount"]))/np.std(data["Credit amount"])  #np.log(data["Credit amount"])
    
    data["Duration"] = (data["Duration"] - np.mean(data["Duration"]))/np.std(data["Duration"])  #np.log(data["Credit amount"])
    
    data["Age"] = (data["Age"] - np.mean(data["Age"]))/np.std(data["Age"])  #np.log(data["Credit amount"])

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
    S = data["Sex_male"].values

    data.to_csv('preprocessing/german.csv')
    if not include_s:
        del data["Sex_male"]

    X = data.drop("Risk_good", 1).values
    y = data["Risk_good"].values
    
    return np.array(X), np.array(y), np.array(S), data


def generate_features_values(prefix, size, index=1):
    a = np.arange(index, size+index)
    return [prefix+str(i) for i in a]
