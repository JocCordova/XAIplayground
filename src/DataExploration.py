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
    # Loads a csv file as a data frame and returns it

    return pd.read_csv(csv_file)


def _create_graphs_dir(path):
    # Creates "Graphs" dir

    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def _get_model_name(estimator, delim="("):
    # Reads model name and/or type from estimator

    text = str(estimator)

    model_name = text.split(delim)[0]

    # If Adaboost add type of weak learner
    if model_name == "AdaBoostClassifier":
        model_name = model_name + "_" + text.split(delim)[1].split("=")[1]

    return model_name


def _sort_class_column(df):
    # Sorts the dataframe by the "Class" column

    categories = ["L", "M", "H"]

    df["Class"] = pd.Categorical(df["Class"], categories=categories)
    df.sort_values(by="Class")

    return df


class Plotter:
    """Plots related to Data visualization

    Attributes
    ----------
    df : pandas df
        loaded pandas df
    path : str
        dir where plots are saved
    """

    def __init__(self, data=FILE_NAME, path=GRAPH_PATH):
        """Loads data and path to be used

        Parameters
        ----------
        data : str, default="\\data\\xAPI-Edu-Data-Edited.csv"
            path to csv file to be loaded as a dataframe
        path : str, default=""\\graphs""
            dir where plots are saved
        """

        self.df = _load_dataset(data)
        self.path = _create_graphs_dir(path)

    def plot_column(self, column, savefig=True):
        """plots single column

        Parameters
        ----------
        column : str
            column to plot
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """

        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.catplot(y=column, kind="count", data=df)
        ax.set(ylabel=column)

        if savefig:
            ax.savefig(path + "/bar_plot_" + str(column) + ".png")

        plt.show()

    def plot_column_grouped(self, x_column, c_column, savefig=True):
        """bar_plots one column grouped on a second column

        Parameters
        ----------
        x_column : str
            column to plot
        c_column : str
            column to group on
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """

        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.catplot(y=x_column, hue=c_column, kind="count", data=df)
        ax.set(xlabel=x_column)

        if savefig:
            ax.savefig(path + "/bar_plot_" + str(x_column) + "_" + str(c_column) + ".png")

        plt.show()

    def hist_plot_column_grouped(self, x_column, c_column="Class", savefig=True):
        """hist_plots one column grouped on a second column

        Parameters
        ----------
        x_column : str
            column to plot
        c_column : str, default="Class"
            column to group on
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """

        df = _sort_class_column(self.df)
        path = self.path

        ax = sns.histplot(data=df, x=x_column, hue=c_column, kde=True, legend=True)
        ax.set(xlabel=x_column)

        if savefig:
            ax.figure.savefig(path + "/hist_plot_" + str(x_column) + "_" + str(c_column) + ".png")

        plt.show()


class ModelPlotter:
    """Plots related to the model itself

    Attributes
    ----------
    path : str
        dir where plots are saved
    """


    def __init__(self, path=GRAPH_PATH):
        """Loads data and path to be used

        Parameters
        ----------
        path : str, default=""\\graphs""
            dir where plots are saved
        """

        self.path = _create_graphs_dir(path)

    def plot_pca(self, var_pca, sum_eigenvalues, scaler_type, pca_threshold=None, savefig=True):
        """Plots pca values, commulative and individual

        Parameters
        ----------
        var_pca : ndarray
            variance of the pca
        sum_eigenvalues : ndarray
            sum of the pca Eigenvalues
        scaler_type : str
            scaler type (for plot names)
        pca_threshold : float range(0,1), optional
            threshold to plot vertical line at
        savefig : bool, default=True
            specifies if plot should be saved as .png
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
        """Plots confusion matrix

        Parameters
        ----------
        cf_matrix : pandas df
            confusion matrix to be plotted
        estimator : estimator
            estimator used (for plot names)
        scaler_type : str
            scaler type (for plot names)
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """

        path = self.path

        model_name = _get_model_name(estimator)

        fig_dims = (10, 5)
        fig, ax = plt.subplots(figsize=fig_dims)

        ax = sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt=".2%", cmap="Blues")
        ax.set(xlabel="Predicted Labels", ylabel="True Labels", title=(scaler_type + "_" + model_name))

        if savefig:
            ax.figure.savefig(f"{path}/confusion_matrix_{scaler_type}_{model_name}.png")

        plt.show()
