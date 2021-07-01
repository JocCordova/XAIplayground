import os
import pickle
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

MODELS_PATH = os.path.dirname(os.getcwd()) + "\\models"


def _get_model_type(estimator, delim="("):
    # Reads model name and/or type from estimator

    text = str(estimator)

    model_name = text.split(delim)[0]

    # If Adaboost add type of weak learner
    if model_name == "AdaBoostClassifier":
        model_name = model_name + "_" + text.split(delim)[1].split("=")[1]

    return model_name


class ModelTuning:
    """Creates and Tunes Models

    Attributes
    ----------
    X_train : ndarray
        Training data
    y_train : ndarray
        Training labels
    X_test : ndarray
        Testing data
    y_test : ndarray
        Testing labels
    X_val : ndarray
        Validation data
    y_val : ndarray
        Validation labels
    """

    def __init__(self, X_data, y_data, train_size, random_state):
        """Splits the data into Train, Test, Val (Test and Val are always the same size)

        Parameters
        ----------
        X_data : pandas df
            Features
        y_data: pandas df
            Target
        train_size : float range(0,1)
            Percentage for train
        random_state : int
            Random state
        """

        self.X_train, X_test_val, self.y_train, y_test_val = train_test_split(
            X_data, y_data, train_size=train_size, random_state=random_state)

        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            X_test_val, y_test_val, test_size=0.5, random_state=random_state)

    def create_weak_learner(self, random_state, verbose, model_type="svm"):
        """Creates weak learner of type dt/svm and cross validates it with the training data

        Parameters
        ----------
        model_type : str
            model type to be created ('dt'/'svm')
        random_state : int
            Random state
        verbose : int
            verbosity level

        Returns
        ----------
        estimator
            fitted weak learner estimator
        """
        if model_type == "dt":
            clf = DecisionTreeClassifier(random_state=random_state)
        if model_type == "svm":
            clf = SVC(probability=True, class_weight=None, random_state=random_state)

        scores = cross_validate(clf, self.X_train, self.y_train, cv=5, n_jobs=-1, scoring=SCORING, verbose=verbose)
        clf.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"avg. {clf} scores:")
            print(f"accuracy: {scores['test_accuracy'].mean()}")
            print(f"precision: {scores['test_precision'].mean()}")
            print(f"recall: {scores['test_recall'].mean()}")
            print(f"f1 score: {scores['test_f1_macro'].mean()}\n")

        return clf

    def create_random_forest(self, random_state, verbose):
        """Creates a random forest estimator and cross validates it with the training data

        Parameters
        ----------
        random_state : int
            Random state
        verbose : int
            verbosity level

        Returns
        ----------
        estimator
            fitted estimator
        """
        clf = RandomForestClassifier(random_state=random_state)

        scores = cross_validate(clf, self.X_train, self.y_train, cv=5, n_jobs=-1, scoring=SCORING, verbose=verbose)
        clf.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"avg. {clf} scores:")
            print(f"accuracy: {scores['test_accuracy'].mean()}")
            print(f"precision: {scores['test_precision'].mean()}")
            print(f"recall: {scores['test_recall'].mean()}")
            print(f"f1 score: {scores['test_f1_macro'].mean()}\n")

        return clf

    def tune_hyperparam(self, estimator, random_state, verbose):
        """Tunes the hyper parameters of the given estimator

        Parameters
        ----------
        estimator : estimator
            estimator to tune
        random_state : int
            Random state
        verbose : int
            verbosity level

        Returns
        ----------
        estimator
            best tuned estimator
        """

        model_type = _get_model_type(estimator)

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
                                 random_state=random_state, n_iter=5, n_jobs=-1, cv=5,
                                 verbose=verbose, refit='f1_macro')

        clf.fit(self.X_test, self.y_test)

        if verbose > 0:
            print(f"Best {model_type} estimator:")
            print(clf.best_estimator_)
            print(f"F1 Score: {clf.best_score_}\n")

        return clf.best_estimator_

    def boost_weak_learners(self, estimator, random_state, verbose):
        """Boosts the estimator with Ada Boost

        Parameters
        ----------
        estimator : estimator
            estimator to boost
        random_state : int
            Random state
        verbose : int
            verbosity level

        Returns
        ----------
        estimator
            best boosted estimator
        """

        ada_clf = AdaBoostClassifier(base_estimator=estimator, random_state=random_state)

        param_distributions = {"learning_rate": [0.0001, 0.001, 0.01, 0.1], "n_estimators": [1, 10, 100, 500]}

        clf = RandomizedSearchCV(estimator=ada_clf, param_distributions=param_distributions,
                                               random_state=random_state, n_iter=5, n_jobs=-1, cv=5,
                                               verbose=verbose, refit='f1_macro')

        clf.fit(self.X_train, self.y_train)

        if verbose > 0:
            print(f"Best Adaboost estimator:")
            print(clf.best_estimator_)
            print(f"F1 Score: {clf.best_score_}\n")

        return clf.best_estimator_

    def get_train_set(self):
        """
        Training set getter
        """
        X_train = self.X_train
        y_train = self.y_train

        return X_train, y_train

    def get_test_set(self):
        """
        Testing set getter
        """
        X_test = self.X_test
        y_test = self.y_test

        return X_test, y_test

    def get_validation_set(self):
        """
        Validation set getter
        """
        X_val = self.X_val
        y_val = self.y_val

        return X_val, y_val


class ModelValidating:
    """Validates the given estimator with different metrics

    Attributes
    ----------
    clf : estimator
        estimator to validate
    X_data : ndarray
        Validation data
    y_data : ndarray
         Validation labels
    y_labels : {ndarray, sparse matrix}
        decoded labels
    scaler : str
        scaler used ('standard'/'min_max') (for plots)
    """


    def __init__(self, estimator, X_data, y_data, y_labels, scaler="standard"):
        """
        Parameters
        ----------
        estimator : estimator
            estimator to validate
        X_data : ndarray
            Validation data
        y_data : ndarray
            Validation labels
        y_labels : {ndarray, sparse matrix}
            decoded labels
        scaler : str
            scaler used ('standard'/'min_max') (for plots)
        """
        self.clf = estimator
        self.X_data = X_data
        self.y_data = y_data
        self.y_labels = y_labels
        self.scaler = scaler

    def plot_confusion_matrix(self, savefig=True):
        """Plots confusion matrix

        Parameters
        ----------
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data
        y_labels = self.y_labels
        scaler = self.scaler

        y_pred = model.predict(X_data)

        cm = confusion_matrix(y_data, y_pred, labels=[1,2,0], normalize='pred')
        cm_df = pd.DataFrame(cm, index=y_labels, columns=y_labels)

        md = ModelPlotter()
        md.plot_confusion_matrix(cm_df, model, scaler, savefig=savefig)

    def classification_report(self):
        """Generates and returns classification report

        Returns
        ----------
        report : str / dict

        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data
        y_labels = self.y_labels

        y_pred = model.predict(X_data)

        return classification_report(y_data, y_pred, target_names=y_labels)

    def get_scores(self):
        """Generates and returns accuracy and f1 score

        Returns
        ----------
        scores : list, length=2
                list of scores [accuracy, f1 score]
        """
        model = self.clf
        X_data = self.X_data
        y_data = self.y_data

        y_pred = model.predict(X_data)

        accuracy = accuracy_score(y_data, y_pred)
        f1 = f1_score(y_data, y_pred, average='macro')

        return list((accuracy, f1))


def save_file(model, file_name=None, path=MODELS_PATH, suffix=None):
    """Saves estimator

    Parameters
    ----------
    file_name : str, optional
        name for the file if None model will be saved as its type
    path : str, default="\\models"
        dir where model is saved
    suffix : str, optional
        suffix to be added
    """
    if file_name is None:
        file_name = _get_model_type(model)

    if suffix is not None:
        file_name = file_name + "_" + str(suffix)

    file = str(path) + "\\" + str(file_name)
    pickle.dump(model, open(file, 'wb'))

    print(f"File Saved as: {file_name}")


def load_file(file_name, path=MODELS_PATH):
    """Loads model

    Parameters
    ----------
    file_name : str
        model to be loaded
    path : str, default="\\models"
        dir where model is saved

    Returns
    ----------
    model : estimator
        loaded model
    """

    file = str(path) + "\\" + str(file_name)

    model = pickle.load(open(file, 'rb'))
    print(f"Model Loaded: {file_name}")

    return model
