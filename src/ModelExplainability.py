import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(1, os.path.dirname(os.getcwd()) + "\\pipeline")
from DataPipeline import Predictor

GRAPH_PATH = os.path.dirname(os.getcwd()) + "\\graphs\\explainer"
CATEGORICAL_FEATURE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15]


def _get_model_name(estimator, delim="("):
    # Reads model name and/or type from estimator

    text = str(estimator)

    model_name = text.split(delim)[0]

    # If Adaboost add type of weak learner
    if model_name == "AdaBoostClassifier":
        model_name = model_name + "_" + text.split(delim)[1].split("=")[1]

    return model_name


def _create_explainer_dir(path):
    # Creates "explainer" dir

    Path(path).mkdir(parents=True, exist_ok=True)

    return path


class TabularExplainer:
    """Tabular explainer for predictions

    Attributes
    ----------
    X_val : ndarray
        features to predict from
    labels : {ndarray, sparse matrix}
        decoded labels
    clf : list of estimator
        estimators to explain
    pipeline : pipeline
        pipeline used
    path : str
        dir where explanations are saved
    explainer : explainer object
        Lime Tabular Explainer object
    """

    def __init__(self, classifier, pipeline, X_train, y_train, X_val, y_labels, path=GRAPH_PATH):
        """Creates Tabular Explainer

        Parameters
        ----------
        classifier : list of estimator
            estimators to explain
        pipeline : pipeline
            pipeline used
        X_train : ndarray
            training data
        y_train : ndarray
            training labels
        X_val : ndarray
            features to predict from
        y_labels : {ndarray, sparse matrix}
            decoded labels
        path : str, default="\\graphs\\explainer"
            dir where explanations are saved
        """
        self.X_val = X_val
        self.labels = y_labels
        self.clf = classifier
        self.pipeline = pipeline
        self.path = _create_explainer_dir(path)

        self.explainer = LimeTabularExplainer(training_data=np.array(X_train), training_labels=y_train,
                                              feature_names=X_train.columns, class_names=y_labels)

    def explain(self, clf_index=0, data_index=0, num_features=15, savefig=True):
        """Explains instance with a given estimator

        Parameters
        ----------
        clf_index : int, default=0
            index of estimator to explain
        data_index : int, default=0
            index of prediction to explain
        num_features : int, default=15
            number of features to show
        savefig : bool, default=True
            specifies if Explanation should be saved
        """
        X_val = self.X_val
        clf = self.clf[clf_index]
        pipeline = self.pipeline
        path = self.path

        X_val = X_val.reset_index(drop=True)

        # Create pipeline object
        pred = Predictor(clf, pipeline)

        # Get names for the plots
        model_name = str(_get_model_name(clf))
        scaler = pipeline.get_scaler_type()

        # Explain instance
        exp = self.explainer.explain_instance(X_val.iloc[data_index], pred.predict_proba, num_features=num_features,
                                              top_labels=3)

        label = list((X_val.iloc[data_index], X_val.iloc[data_index]))

        fig = exp.as_pyplot_figure(pred.predict(label)[0])
        fig.set_size_inches(18.5, 10.5)

        plt.show()

        if savefig:
            exp.save_to_file(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + ".html")
            fig.figure.savefig(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + ".png")
