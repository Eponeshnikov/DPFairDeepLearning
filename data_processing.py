import numpy as np
import pandas as pd


def preprocessing_adult_1(df, use_age=False, include_s=True):
    n_data = len(df)
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
    # df = df.merge(pd.get_dummies(df['Education-num'], drop_first=True, prefix='Education-num'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.Outcome, drop_first=True,
                                 prefix='outcome'), left_index=True, right_index=True)

    # feature scaling
    # apply stander scaler
    # df.Age = (df.Age - np.mean(df.Age))/np.std(df.Age)

    df.fnlwgt = (df.fnlwgt - np.mean(df.fnlwgt)) / np.std(df.fnlwgt)

    df['Capital-gain'] = (df['Capital-gain'] -
                          np.mean(df['Capital-gain'])) / np.std(df['Capital-gain'])
    df['Capital-loss'] = (df['Capital-loss'] -
                          np.mean(df['Capital-loss'])) / np.std(df['Capital-loss'])

    # apply min max scaler
    df['hours-per-week'] = (df['hours-per-week'] - np.min(df['hours-per-week'])) / (
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


def preprocessing_adult_2(target, sensitive, clr_ratio=None):
    # make noNULL file with: grep -v NULL rawdata_mkmk01.csv | cut -f1,3,4,6- -d, > rawdata_mkmk01_noNULL.csv
    EPS = 1e-8

    UNPRIVILEGED_ARR = np.array([1, 0])
    PRIVILEGED_ARR = np.array([0, 1])
    SENSATTR_RULES = {
        "sex": lambda x, thresh1, thresh2: UNPRIVILEGED_ARR
        if x == "Female"
        else PRIVILEGED_ARR,
        "age": lambda x, thresh1, thresh2: UNPRIVILEGED_ARR
        if thresh1 <= int(x) < thresh2
        else PRIVILEGED_ARR,
    }

    def binarize_sensattr(val, sensattr, thresh1, thresh2):
        return SENSATTR_RULES[sensattr](val, thresh1, thresh2)

    def bucket(x, buckets):
        x = float(x)
        n = len(buckets)
        label = n
        for i in range(len(buckets)):
            if x <= buckets[i]:
                label = i
                break
        template = [0.0 for j in range(n + 1)]
        template[label] = 1.0
        return template

    def onehot(x, choices):
        if not x in choices:
            print('could not find "{}" in choices'.format(x))
            print(choices)
            raise Exception()
        label = choices.index(x)
        template = [0.0 for j in range(len(choices))]
        template[label] = 1.0
        return template

    def continuous(x):
        return [float(x)]

    def parse_row(row, headers, headers_use, fns, sensitive, target, thresh1, thresh2):
        new_row_dict = {}
        for i in range(len(row)):
            x = row[i]
            hdr = headers[i]
            new_row_dict[hdr] = fns[hdr](x)
            if hdr == sensitive:
                sens_att = binarize_sensattr(x, hdr, thresh1, thresh2)

        label = new_row_dict[target]
        new_row = []
        for h in headers_use:
            new_row = new_row + new_row_dict[h]
        return new_row, label, sens_att

    def whiten(X, mn, std):
        mntile = np.tile(mn, (X.shape[0], 1))
        stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
        X = X - mntile
        X = np.divide(X, stdtile)
        return X

    thresh1, thresh2 = 0, 0
    if sensitive == "age":
        if clr_ratio == [0.5, 0.5]:
            thresh1, thresh2 = 25, 44
        elif clr_ratio == [0.66, 0.33]:
            thresh1, thresh2 = 38, 60
        elif clr_ratio == [0.33, 0.33]:
            thresh1, thresh2 = 28, 40
        elif clr_ratio == [0.1, 0.1]:
            thresh1, thresh2 = 32, 36
        elif clr_ratio == [0.06, 0.36]:
            thresh1, thresh2 = 0, 30
        elif clr_ratio == [0.01, 0.01]:
            thresh1, thresh2 = 71, 100

    f_in_tr = "./dataset/adult.data"
    f_in_te = "./dataset/adult.test"

    REMOVE_MISSING = True
    MISSING_TOKEN = "?"

    headers = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex," \
              "capital-gain,capital-loss,hours-per-week,native-country,income".split(",")
    headers_use = "age,workclass,education,education-num,marital-status,occupation,relationship,race,capital-gain," \
                  "capital-loss,hours-per-week,native-country".split(",")

    options = {
        "age": "buckets",
        "workclass": "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, "
                     "Never-worked",
        "fnlwgt": "continuous",
        "education": "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, "
                     "Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",
        "education-num": "continuous",
        "marital-status": "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, "
                          "Married-AF-spouse",
        "occupation": "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, "
                      "Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, "
                      "Priv-house-serv, Protective-serv, Armed-Forces",
        "relationship": "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",
        "race": "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",
        "sex": "Female, Male",
        "capital-gain": "continuous",
        "capital-loss": "continuous",
        "hours-per-week": "continuous",
        "native-country": "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US("
                          "Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, "
                          "Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, "
                          "Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, "
                          "Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands",
        "income": " <=50K,>50K",
    }

    buckets = {"age": [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]}

    options = {k: [s.strip() for s in sorted(options[k].split(","))]
               for k in options}

    #print(f"OPTIONS = {options}")
    fns = {
        "age": lambda x: bucket(x, buckets["age"]),
        "workclass": lambda x: onehot(x, options["workclass"]),
        "fnlwgt": lambda x: continuous(x),
        "education": lambda x: onehot(x, options["education"]),
        "education-num": lambda x: continuous(x),
        "marital-status": lambda x: onehot(x, options["marital-status"]),
        "occupation": lambda x: onehot(x, options["occupation"]),
        "relationship": lambda x: onehot(x, options["relationship"]),
        "race": lambda x: onehot(x, options["race"]),
        "sex": lambda x: onehot(x, options["sex"]),
        "capital-gain": lambda x: continuous(x),
        "capital-loss": lambda x: continuous(x),
        "hours-per-week": lambda x: continuous(x),
        "native-country": lambda x: onehot(x, options["native-country"]),
        "income": lambda x: onehot(x.strip("."), options["income"]),
    }

    D = {}
    for f, phase in [(f_in_tr, "training"), (f_in_te, "test")]:
        dat = [s.strip().split(",") for s in open(f, "r").readlines()]

        X = []
        Y = []
        A = []
        # print(phase)

        for r in dat:
            row = [s.strip() for s in r]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([""], ["|1x3 Cross validator"]):
                continue
            newrow, label, sens_att = parse_row(
                row, headers, headers_use, fns, sensitive, target, thresh1, thresh2
            )
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        npA = np.expand_dims(npA[:, 1], 1)

        D[phase] = {}
        D[phase]["X"] = npX
        D[phase]["Y"] = npY
        D[phase]["A"] = npA

        # print(npX.shape)
        # print(npY.shape)
        # print(npA.shape)

    # should do normalization and centring
    mn = np.mean(D["training"]["X"], axis=0)
    std = np.std(D["training"]["X"], axis=0)
    # print(mn, std)
    D["training"]["X"] = whiten(D["training"]["X"], mn, std)
    D["test"]["X"] = whiten(D["test"]["X"], mn, std)

    n = D["training"]["X"].shape[0]
    unshuf = np.arange(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = unshuf[:valid_ct]
    train_inds = unshuf[valid_ct:]

    D["training"]["X"] = D["training"]["X"].astype(np.float32).squeeze()
    D["training"]["Y"] = np.argmax(
        D["training"]["Y"].astype(np.float32), axis=1
    ).squeeze()
    D["training"]["A"] = D["training"]["A"].astype(np.float32).squeeze()

    D["test"]["X"] = D["test"]["X"].astype(np.float32).squeeze()
    D["test"]["Y"] = np.argmax(
        D["test"]["Y"].astype(np.float32), axis=1).squeeze()
    D["test"]["A"] = D["test"]["A"].astype(np.float32).squeeze()

    total_p = sum(D["test"]["A"])
    total_up = D["test"]["A"].shape[0] - sum(D["test"]["A"])

    total_e = sum(D["test"]["Y"])
    total_ie = D["test"]["Y"].shape[0] - sum(D["test"]["Y"])

    priv_elg = sum(D["test"]["Y"] * D["test"]["A"])
    priv_inelg = total_p - priv_elg
    unpriv_elg = total_e - priv_elg
    unpriv_inelg = total_ie - priv_inelg

    min_num = int(min(priv_elg, priv_inelg, unpriv_elg, unpriv_inelg))

    # print("Privileged eligible", priv_elg)
    # print("Privileged ineligible", priv_inelg)
    # print("Unprivileged eligible", unpriv_elg)
    # print("Unprivileged ineligible", unpriv_inelg)

    priv_elg_inds = D["test"]["Y"] * D["test"]["A"] > 0
    priv_elg_test_X = D["test"]["X"][priv_elg_inds][0:min_num]
    priv_elg_test_Y = D["test"]["Y"][priv_elg_inds][0:min_num]
    priv_elg_test_A = D["test"]["A"][priv_elg_inds][0:min_num]

    priv_inelg_inds = (1 - D["test"]["Y"]) * D["test"]["A"] > 0
    priv_inelg_test_X = D["test"]["X"][priv_inelg_inds][0:min_num]
    priv_inelg_test_Y = D["test"]["Y"][priv_inelg_inds][0:min_num]
    priv_inelg_test_A = D["test"]["A"][priv_inelg_inds][0:min_num]

    unpriv_elg_inds = D["test"]["Y"] * (1 - D["test"]["A"]) > 0
    unpriv_elg_test_X = D["test"]["X"][unpriv_elg_inds][0:min_num]
    unpriv_elg_test_Y = D["test"]["Y"][unpriv_elg_inds][0:min_num]
    unpriv_elg_test_A = D["test"]["A"][unpriv_elg_inds][0:min_num]

    unpriv_inelg_inds = (1 - D["test"]["Y"]) * (1 - D["test"]["A"]) > 0
    unpriv_inelg_test_X = D["test"]["X"][unpriv_inelg_inds][0:min_num]
    unpriv_inelg_test_Y = D["test"]["Y"][unpriv_inelg_inds][0:min_num]
    unpriv_inelg_test_A = D["test"]["A"][unpriv_inelg_inds][0:min_num]

    #print(
    #    D["test"]["Y"][priv_elg_inds].shape[0],
    #    D["test"]["Y"][priv_inelg_inds].shape[0],
    #    D["test"]["Y"][unpriv_elg_inds].shape[0],
    #    D["test"]["Y"][unpriv_inelg_inds].shape[0],
    #)

    D["test"]["X"] = np.concatenate(
        (priv_elg_test_X, priv_inelg_test_X,
         unpriv_elg_test_X, unpriv_inelg_test_X),
        axis=0,
    )
    D["test"]["Y"] = np.concatenate(
        (priv_elg_test_Y, priv_inelg_test_Y,
         unpriv_elg_test_Y, unpriv_inelg_test_Y),
        axis=0,
    )
    D["test"]["A"] = np.concatenate(
        (priv_elg_test_A, priv_inelg_test_A,
         unpriv_elg_test_A, unpriv_inelg_test_A),
        axis=0,
    )

    data_dict = {
        "x_train": D["training"]["X"],
        "x_test": D["test"]["X"],
        "y_train": D["training"]["Y"],
        "y_test": D["test"]["Y"],
        "attr_train": D["training"]["A"],
        "attr_test": D["test"]["A"],
        "train_inds": train_inds,
        "valid_inds": valid_inds,
    }
    return data_dict


def preprocessing_german(data, include_s=True):
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
    S = data["Sex_male"].values

    data.to_csv('preprocessing/german.csv')
    if not include_s:
        del data["Sex_male"]

    X = data.drop("Risk_good", 1).values
    y = data["Risk_good"].values

    return np.array(X), np.array(y), np.array(S), data


def generate_features_values(prefix, size, index=1):
    a = np.arange(index, size + index)
    return [prefix + str(i) for i in a]
