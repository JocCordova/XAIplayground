import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from DataExploration import ModelPlotter

# Make a dict containing the scorers to be used
SCORING = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro'),
           'recall': make_scorer(recall_score, average='macro'),
           'f1_macro': make_scorer(f1_score, average='macro')}


def _get_model_name(estimator, delim="("):
    """
    Reads model name and/or type from estimator
    :param estimator:(estimator) estimator to extract name from
    :param delim:(str) delimeter to separate
    :return:(str) model name and/or type
    """

    text = str(estimator)

    model_name = text.split(delim)[0]

    # If Adaboost add type of weak learner
    if model_name == "AdaBoostClassifier":
        model_name = text.split(delim)[1].split("=")[1]

    return model_name


class ModelTuning:
    """
    Creates and Tunes Models
    """

    def __init__(self, X_data, y_data, train, random_state):
        """
        Splits the data into Train,Test,Val
        :param X_data: (df) Features
        :param y_data: (df) Target
        :param train: (float) percentage for train
        :param random_state: Random state
        """
        test_val = 1 - train
        self.X_train, X_test_val, self.y_train, y_test_val = train_test_split(
            X_data, y_data, test_size=test_val, random_state=random_state)

        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=random_state)

    def create_weak_learner(self, random_state, verbose, model_type="svm"):
        """
        Creates weak learner of type dt/svm
        :param model_type: (string) model type to be created ('dt'/'svm')
        :param random_state: Random state
        :param verbose: verbosity level
        :return: fitted weak learner classifier
        """
        if model_type == "dt":
            clf = DecisionTreeClassifier(random_state=random_state)
            print(_get_model_name(clf))
        if model_type == "svm":
            clf = SVC(probability=True, class_weight=None, random_state=random_state)
            print(_get_model_name(clf))

        scores = cross_validate(clf, self.X_train, self.y_train, cv=5, n_jobs=-1, scoring=SCORING, verbose=verbose)
        clf.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"avg. {clf} scores:")
            print(f"test_accuracy: {scores['test_accuracy'].mean()}")
            print(f"test_precision: {scores['test_precision'].mean()}")
            print(f"test_recall: {scores['test_recall'].mean()}")
            print(f"test_f1_macro: {scores['test_f1_macro'].mean()}\n")

        return clf

    def create_random_forest(self, random_state, verbose):
        """
        Creates weak learner of type dt/svm
        :param random_state: Random state
        :param verbose: verbosity level
        :return: fitted random forest classifier
        """
        clf = RandomForestClassifier(random_state=random_state)
        print(_get_model_name(clf))

        scores = cross_validate(clf, self.X_train, self.y_train, cv=5, n_jobs=-1, scoring=SCORING, verbose=verbose)
        clf.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"avg. {clf} scores:")
            print(f"test_accuracy: {scores['test_accuracy'].mean()}")
            print(f"test_precision: {scores['test_precision'].mean()}")
            print(f"test_recall: {scores['test_recall'].mean()}")
            print(f"test_f1_macro: {scores['test_f1_macro'].mean()}\n")

        return clf

    def tune_hyperparam(self, estimator, random_state, verbose):
        """
        Tunes the hyper parameters of the given estimator
        :param estimator: estimator to tune
        :param random_state: Random state
        :param verbose: verbosity level
        :return: fitted and tuned estimator
        """

        model_type = _get_model_name(estimator)

        if model_type == "DecisionTreeClassifier":
            param_distributions = {"max_features": ["auto", "sqrt", "log2", None], "max_depth": range(1, 15)}
        if model_type == "SVC":
            param_distributions = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                    'C': [1, 10, 100, 1000]},
                                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        if model_type == "RandomForestClassifier":
            param_distributions = {"n_estimators": [1, 10, 50, 100, 200, 500],
                                   "max_features": ["auto", "sqrt", "log2", None],
                                   "max_depth": range(1, 15)}

        clf = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions,
                                 random_state=random_state, n_iter=10, n_jobs=-1, cv=5,
                                 verbose=verbose, refit='f1_macro')

        clf.fit(self.X_test, self.y_test)

        if verbose > 0:
            print(f"Best {model_type} estimator:")
            print(clf.best_estimator_)
            print(f"F1 Score: {clf.best_score_}\n")

        return clf.best_estimator_

    def boost_weak_learners(self, estimator, random_state, verbose):
        """
        Boosts the estimator with Ada Boost
        :param estimator: estimator to boost
        :param random_state: Random state
        :param verbose: verbosity level
        :return: fitted and tuned estimator
        """

        ada_clf = AdaBoostClassifier(base_estimator=estimator, random_state=random_state)

        param_distributions = {"learning_rate": [0.0001, 0.001, 0.01, 0.1], "n_estimators": [1, 10, 100, 500]}

        rnd_search_cv_ada = RandomizedSearchCV(estimator=ada_clf, param_distributions=param_distributions,
                                               random_state=42, n_iter=5, n_jobs=-1, cv=5,
                                               verbose=verbose, refit='f1_macro')

        rnd_search_cv_ada.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"Best Adaboost estimator:")
            print(rnd_search_cv_ada.best_estimator_)
            print(f"F1 Score: {rnd_search_cv_ada.best_score_}\n")

        return rnd_search_cv_ada.best_estimator_

    def get_train_set(self):
        """
        Training set getter
        :return: (X_train, y_train)
        """
        X_train = self.X_train
        y_train = self.y_train

        return X_train, y_train

    def get_test_set(self):
        """
        Training set getter
        :return: (X_test, y_test)
        """
        X_test = self.X_test
        y_test = self.y_test

        return X_test, y_test

    def get_validation_set(self):
        """
        Training set getter
        :return: (X_val, y_val)
        """
        X_val = self.X_val
        y_val = self.y_val

        return X_val, y_val


class ModelValidating:
    """
    Validates the given classifier with different metrics
    """

    def __init__(self, classifier, X_data, y_data, y_labels, scaler="standard"):
        """
        :param classifier: classifier to validate
        :param X_data: features
        :param y_data: target
        :param y_labels: target labels
        :param scaler: scaler used (for plots)
        """
        self.clf = classifier
        self.X_data = X_data
        self.y_data = y_data
        self.y_labels = y_labels
        self.scaler = scaler

    def plot_confusion_matrix(self, savefig=True):
        """
        plots confusion matrix
        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data
        y_labels = self.y_labels
        scaler = self.scaler

        y_pred = model.predict(X_data)

        cm = confusion_matrix(y_data, y_pred, normalize='pred')
        cm_df = pd.DataFrame(cm, index=y_labels, columns=y_labels)

        md = ModelPlotter()
        md.plot_confusion_matrix(cm_df, model, scaler, savefig=savefig)

    def classification_report(self):
        """
        returns classification report
        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data
        y_labels = self.y_labels

        y_pred = model.predict(X_data)

        return classification_report(y_data, y_pred, target_names=y_labels)

    def get_scores(self):
        """
        returns classification report
        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data

        y_pred = model.predict(X_data)

        accuracy = accuracy_score(y_data, y_pred)
        f1 = f1_score(y_data, y_pred, average='macro')
        return list((accuracy, f1))




