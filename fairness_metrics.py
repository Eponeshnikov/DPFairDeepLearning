import numpy as np
from sklearn.metrics import accuracy_score


def statistical_parity_score(y_pred, s):
    """ This measure the proportion of positive and negative class in protected and non protected group """

    alpha_1 = np.sum(np.logical_and(y_pred == 1, s == 1)) / \
        float(np.sum(s == 1))
    beta_1 = np.sum(np.logical_and(y_pred == 1, s == 0)) / \
        float(np.sum(s == 0))
    return np.abs(alpha_1 - beta_1)


def confusion_matrix_score(y_pred, y_true, s):
    """
        Parameters
        ----------
        y_pred : 1-D array size n
            Label returned by the model
        y_true : 1-D array size n
            Real label
            # print("Training %s"%(name))
        s: 1-D size n protected attribut
        Return
        -------
        equal_opportunity True positive error rate across group
        equal_disadvantage False positive error rate across group
    """

    alpha_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))) / float(
        np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))) / float(np.sum(
        np.logical_and(y_true == 1, s == 1)))

    alpha_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 0))) / float(np.sum(
        np.logical_and(y_true == 0, s == 0)))
    beta_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 1))) / float(np.sum(
        np.logical_and(y_true == 0, s == 1)))

    equal_opportunity = np.abs(alpha_1 - beta_1)
    equal_disadvantage = np.abs(alpha_2 - beta_2)
    return equal_opportunity, equal_disadvantage


def cross_val_fair_score(model, X, y, cv, protected_attrib, scoring='statistical_parity_score'):
    """
    Kfold cross validation method, that returns the accuracy, fairness performences

    Parameters
    ----------

    model : class with fit and predict methods
    X: features matrice
    y: labels
    cv: Kfold cross validation from Sklearn
    protected_attrib: Protected attribute
    scoring : str, default = "statistical_parity_score"
        Possible inputs are:
        'statistical_parity_score' match the selection rate across groups
        'equalized_odds' match true positive rates across groups
        'equal_opportunity' match true positive and false positive rates across groups

    """
    scores = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if scoring == 'statistical_parity_score':
            score = statistical_parity_score(
                y_pred, protected_attrib)
            scores.append(score)
        elif scoring == 'equalized_odds':
            bias1, bias2 = confusion_matrix_score(
                y_pred, y_test, protected_attrib)
            scores.append(bias1 + bias2)
        elif scoring == 'equal_opportunity':
            score, _ = confusion_matrix_score(
                y_pred, y_test, protected_attrib)
            scores.append(score)
    return np.array(scores)


def cross_val_fair_scores(model, X, y, cv, protected_attrib, fit_sensitive=False):
    """
    model : class with fit and predict methods
    X: features matrices
    y: labels
    cv: Kfold cross validation from Sklearn
    protected_attrib: Protected attribute
    scoring : "statistical_parity_score" | "equalized_odds" | "equal_opportunity"
    fit_sensitive: True if the fit method receive sensitive attribute. Only for fairness-aware estimators
    """
    st_scores = []
    equal_odds = []
    equal_opps = []
    accuracy = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s_train, s_test = protected_attrib[train_index], protected_attrib[test_index]
        if fit_sensitive:
            clf = model.fit(X_train, y_train, sensitive_features=s_train)
            y_pred = clf.predict(X_test, sensitive_features=s_test)
        else:
            clf = model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        st_score = statistical_parity_score(y_pred, s_test)
        st_scores.append(st_score)

        tpr, fpr = confusion_matrix_score(y_pred, y_test, s_test)
        equal_odds.append(tpr + fpr)

        tpr, _ = confusion_matrix_score(y_pred, y_test, s_test)
        equal_opps.append(tpr)

        accuracy.append(accuracy_score(y_test, y_pred))
    return accuracy, st_scores, equal_odds, equal_opps
