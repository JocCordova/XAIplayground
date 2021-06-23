import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from pathlib import Path

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"
GRAPH_PATH = os.path.dirname(os.getcwd()) + "\\graphs"

sns.set_theme(style="ticks", color_codes=True)


def _load_dataset(csv_file):
    """
    Loads a csv file as a data frame and returns it
    :param csv_file: name of the csv file to load
    :return: csv file as a df
    """
    return pd.read_csv(csv_file)


def _create_graphs_dir(path):
    """
    Creates "Graphs" dir
    :param path: dir to be created
    :return: path to directory
    """

    Path(path).mkdir(parents=True, exist_ok=True)

    return path


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


def _sort_class_column(df):
    """
    Sorts the dataframe by the "Class" column
    :param df:(df) dataframe to sort
    :return:(df) sorted dataframe
    """
    categories = ["L", "M", "H"]

    df["Class"] = pd.Categorical(df["Class"], categories=categories)
    df.sort_values(by="Class")

    return df


class Plotter:
    """
    Plots related to Data visualization

    """

    def __init__(self, data=FILE_NAME, path=GRAPH_PATH):
        """
        Loads data and path to be used
        :param data: (str) path to csv file to be loaded as a dataframe
        :param path: (str) dir to save plots to
        """
        self.df = _load_dataset(data)
        self.path = _create_graphs_dir(path)

    def plot_column(self, column, savefig=True):
        """
        Takes one Column and plots it, and saves it as a .png
        :param column: (str) column to plot
        :param savefig: (bool) specifies if plot should be saved
        """
        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.catplot(y=column, kind="count", data=df)
        ax.set(ylabel=column)

        if savefig:
            ax.savefig(path + "/bar_plot_" + str(column) + ".png")

        plt.show()

    def plot_column_grouped(self, x_column, c_column, savefig=True):
        """
        Takes one Columns and bar_plots it compared with the second column
        :param x_column: (str) column to plot
        :param c_column: (str) column to group on
        :param savefig: (bool) specifies if plot should be saved
        """

        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.catplot(y=x_column, hue=c_column, kind="count", data=df)
        ax.set(xlabel=x_column)

        if savefig:
            ax.savefig(path + "/bar_plot_" + str(x_column) + "_" + str(c_column) + ".png")

        plt.show()

    def hist_plot_column_grouped(self, x_column, c_column="Class", savefig=True):
        """
        Takes one Column and hist_plots it compared with the second column
        :param x_column: (str) column to plot
        :param c_column: (str) column to group on
        :param savefig: (bool) specifies if plot should be saved
        """

        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.histplot(data=df, x=x_column, hue=c_column, kde=True, legend=True)
        ax.set(xlabel=x_column)

        if savefig:
            ax.figure.savefig(path + "/hist_plot_" + str(x_column) + "_" + str(c_column) + ".png")

        plt.show()


class ModelPlotter:
    """
    Plots related to the model itself
    """

    def __init__(self, path=GRAPH_PATH):
        """
        Loads data and path to be used
        :param path: (str) dir to save plots to
        """
        self.path = _create_graphs_dir(path)

    def plot_pca(self, var_pca, sum_eigenvalues, scaler_type, pca_threshold=None, savefig=True):
        """
        Plots pca values, commulative and individual
        :param var_pca: variance of the pca
        :param sum_eigenvalues: sum of the pca Eigenvalues
        :param scaler_type: (str) scaler type (for plot names)
        :param pca_threshold: (float range(0,1)) threshold to plot vertical line at
        :param savefig: (bool) specifies if plot should be saved
        """
        path = self.path

        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)

        exp_df = pd.DataFrame(zip(var_pca, range(0, len(var_pca))), columns=["exp_var", "index"])
        cum_df = pd.DataFrame(zip(sum_eigenvalues, range(0, len(sum_eigenvalues))), columns=["cum_var", "index"])

        if pca_threshold is not None:
            threshold_index = cum_df.index[cum_df['cum_var'] >= pca_threshold].tolist()
            plt.axvline(threshold_index[0], color='red', alpha=0.5, linestyle="--")

        sns.barplot(x="index", y="exp_var", data=exp_df, ax=ax, label="Individual explained variance")
        sns.lineplot(x="index", y="cum_var", data=cum_df, drawstyle='steps-pre', label="Cumulative explained variance")

        ax.set(ylabel='Explained variance ratio', xlabel='Principal component index',
               title=f"PCA with {scaler_type} scaler")

        if savefig:
            ax.figure.savefig(path + "/pca_plot_" + str(scaler_type) + ".png")

        plt.show()

    def plot_confusion_matrix(self, cf_matrix, estimator, scaler_type, savefig=True):
        """
        Plots confusion matrix
        :param cf_matrix: (cf_matrix) confusion matrix to be used
        :param estimator: (estimator) estimator used (for plot names)
        :param scaler_type: (str) scaler type (for plot names)
        :param savefig: (bool) specifies if plot should be saved
        """
        path = self.path

        model_name = _get_model_name(estimator)

        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)

        # TODO cmap and percentages and sort L M H
        ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), linewidths=1, annot=True, ax=ax, fmt='g', cmap="viridis")
        ax.set(xlabel="Predicted Labels", ylabel="True Labels", title=(scaler_type + "_" + model_name))

        if savefig:
            ax.figure.savefig(f"{path}/confusion_matrix_{scaler_type}_{model_name}.png")

        plt.show()
