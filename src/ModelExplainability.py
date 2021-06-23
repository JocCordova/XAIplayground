import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

GRAPH_PATH = os.path.dirname(os.getcwd()) + "\\graphs\\explainer"

sys.path.insert(1, os.path.dirname(os.getcwd()) + "\\pipeline")
from DataPipeline import Predictor

CATEGORICAL_FEATURE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15]


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
        model_name = model_name + "_" + text.split(delim)[1].split("=")[1]

    return model_name


def _create_explainer_dir(path):
    """
    Creates "explainer" dir
    :param path: dir to be created
    :return: path to directory
    """

    Path(path).mkdir(parents=True, exist_ok=True)

    return path


class TabularExplainer:

    def __init__(self, classifier, pipeline, X_train, y_train, X_val, y_labels, path=GRAPH_PATH):
        self.labels = y_labels
        self.X_val = X_val
        self.path = _create_explainer_dir(path)
        self.clf = classifier
        self.pipeline = pipeline

        self.explainer = LimeTabularExplainer(training_data=np.array(X_train), training_labels=y_train,
                                              feature_names=X_train.columns, class_names=y_labels)

    def explain(self, clf_index=0, data_index=0, num_features=15):
        y_labels = self.labels
        X_val = self.X_val
        clf = self.clf[clf_index]
        path = self.path
        pipeline = self.pipeline

        X_val = X_val.reset_index(drop=True)

        pred = Predictor(clf, pipeline)

        model_name = str(_get_model_name(clf))
        scaler = pipeline.get_scaler_type()

        exp = self.explainer.explain_instance(X_val.iloc[data_index], pred.predict_proba, num_features=num_features,
                                              top_labels=3)

        label = list((X_val.iloc[data_index], X_val.iloc[data_index]))
        exp.save_to_file(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + ".html")

        fig = exp.as_pyplot_figure(pred.predict(label)[0])
        fig.set_size_inches(18.5, 10.5)
        fig.figure.savefig(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + ".jpg")
