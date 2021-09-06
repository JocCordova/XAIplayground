import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import shap
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
    X_train : ndarray
        training data
    y_train : ndarray
        training labels
    X_test : ndarray
        features to predict from
    y_labels : {ndarray, sparse matrix}
        decoded labels
    labels : {ndarray, sparse matrix}
        decoded labels
    clf : list of estimator
        estimators to explain
    pipeline : pipeline
        pipeline used
    path : str
        dir where explanations are saved

    """

    def __init__(self, classifier, pipeline, X_train, y_train, X_test, y_labels, path=GRAPH_PATH):
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
        X_test : ndarray
            features to predict from
        y_labels : {ndarray, sparse matrix}
            decoded labels
        path : str, default="\\graphs\\explainer"
            dir where explanations are saved
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_labels = y_labels
        self.clf = classifier
        self.pipeline = pipeline
        self.path = _create_explainer_dir(path)

    def explain_lime(self, clf_index=0, data_index=0, num_features=8, suffix="", savefig=True):
        """Explains instance with lime with a given estimator

        Parameters
        ----------
        clf_index : int, default=0
            index of estimator to explain
        data_index : int, default=0
            index of prediction to explain
        num_features : int, default=8
            number of features to show
        suffix : str, default=None
            suffix for plot names
        savefig : bool, default=True
            specifies if Explanation should be saved
        """
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_labels = self.y_labels
        clf = self.clf[clf_index]
        pipeline = self.pipeline
        path = self.path

        X_test = X_test.reset_index(drop=True)

        # Create pipeline object
        pred = Predictor(clf, pipeline)

        # Get names for the plots
        model_name = str(_get_model_name(clf))
        scaler = pipeline.get_scaler_type()

        # Create lime explainer
        explainer = LimeTabularExplainer(training_data=np.array(X_train), training_labels=y_train,
                                         feature_names=X_train.columns, class_names=y_labels)

        # Explain instance
        exp = explainer.explain_instance(X_test.iloc[data_index], pred.predict_proba, num_features=num_features,
                                         top_labels=3)
        exp.show_in_notebook(show_table=True, show_all=True)

        label = list((X_test.iloc[data_index], X_test.iloc[data_index]))

        plt.rcParams["axes.titlesize"] = 30
        plt.rcParams["axes.labelsize"] = 28
        plt.rcParams["ytick.labelsize"] = 25
        plt.rcParams["xtick.labelsize"] = 20

        fig = exp.as_pyplot_figure(pred.predict(label)[0])
        fig.set_size_inches(15.5, 10.5)

        plt.tight_layout()

        if savefig:
            exp.save_to_file(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + suffix + ".html")
            plt.savefig(path + "\\lime_" + model_name + "_" + scaler + "_" + str(data_index) + suffix + ".pdf")

        plt.show()

    def explain_shap(self, clf_index=0, n_samples=10, suffix="", class_index=None, multi_class=True, savefig=True):
        """Explains features with shap for with a given estimator

        Parameters
        ----------
        clf_index : int, default=0
            index of estimator to explain
        num_samples : int, default=10
            number of samples to use on explainer
        suffix : str, default=None
            suffix for plot names
        class_index : int or list of int, default=[1,2,0]
            indexes of classes to explain
        multi_class : bool, default=True
            specifies if multi-class explanation should be created
        savefig : bool, default=True
            specifies if Explanation should be saved
        """

        X_train = self.X_train
        y_labels = self.y_labels
        X_test = self.X_test
        clf = self.clf[clf_index]
        pipeline = self.pipeline
        path = self.path

        if class_index is None:
            class_index = [1, 2, 0]

        # Create pipeline object
        pred = Predictor(clf, pipeline)

        # Get names for the plots
        model_name = str(_get_model_name(clf))
        scaler = pipeline.get_scaler_type()

        X_test = X_test.reset_index(drop=True)

        # Create shap explainer

        explainer = shap.KernelExplainer(pred.predict_proba, data=shap.kmeans(X_train, n_samples), link="identity")
        shap_values = explainer.shap_values(X_test)

        fig = plt.figure()
        plt.rcParams["axes.titlesize"] = 30
        plt.rcParams["axes.labelsize"] = 28
        plt.rcParams["ytick.labelsize"] = 25
        plt.rcParams["xtick.labelsize"] = 20

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=25)

        plt.tight_layout()

        def savefig(class_name):
            plt.tight_layout()
            class_name = class_name + suffix

            plt.rcParams["image.cmap"] = 'Pastel1'
            plt.savefig(path + "\\shap_" + model_name + "_" + scaler + "_" + class_name + ".pdf")
            plt.show()

        for i in class_index:
            class_name = y_labels[i]

            shap.summary_plot(shap_values=shap_values[i], features=X_test, feature_names=X_train.columns, show=False,
                              max_display=8)
            plt.title("Global explanation for Class " + str(class_name))

            if savefig:
                savefig("class_" + str(class_name))

        if multi_class:
            shap.summary_plot(shap_values=shap_values, features=X_test, class_names=y_labels, show=False, max_display=8)
            plt.title("Global explanation multi-class")

            if savefig:
                savefig("multiclass")


