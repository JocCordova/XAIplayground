import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import os

from DataExploration import ModelPlotter

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"
BASIC_DECODER = [0, 1, 2]


def _load_dataset(csv_file):
    """
    Loads a csv file as a data frame and returns it
    :param csv_file: name of the csv file to load
    :return: csv file as a df
    """
    return pd.read_csv(csv_file)


class Preprocess:
    """
    Separates columns and target and tunes columns
    """

    def __init__(self, data=FILE_NAME, target="Class"):
        """
        Loads Data and Target to be used
        :param data:
        :param target:
        """
        self.df = _load_dataset(data)
        self.target = self.df.pop(target)
        self.encoder = LabelEncoder()

    def check_missing_values(self):
        """
        Counts all missing row values
        :return: sum of missing values
        """
        df = self.df

        null_cols = df.isnull().any(axis=1).sum()

        return null_cols

    def replace_values(self, column, old_values, new_value):
        """
        Replaces categorical values in a specific column
        :param column: (string) column where the value is
        :param old_values: (string/string[]) value to be changed
        :param new_value: (string/string[]) value to be changed to
        """

        for old_value in old_values:
            self.df.loc[(self.df[column] == old_value), column] = new_value

    def one_hot_encode(self, columns, prefix):
        """
        One Hot Encodes values in a specific column with a specific prefix
        :param columns: column to be encoded
        :param prefix: prefix to be used
        """
        self.df = pd.get_dummies(self.df, columns=columns, prefix=prefix)

    def target_encode(self):
        """
        Encodes target column
        """
        target = self.target

        self.target = self.encoder.fit_transform(target)

    def target_decode(self, target=BASIC_DECODER):
        """
        Decodes list using using target decoder
        :param target: list to be decoded
        """

        return self.encoder.inverse_transform(target)

    def get_data(self):
        """
        Gets df and target
        :return: X df and target df
        """
        df = self.df
        target = self.target

        return df, target


class FeaturePreprocess:
    """
    Scaling and dimensionality reduction
    """

    def __init__(self, X_data, y_data, n_components=15, scaler_type="standard"):
        """
        Creates a pipeline with the selected scalers and reductors
        :param X_data: (df) feature columns
        :param y_data: (df) target column
        :param n_components: (int) number of components to apply pca to
        :param scaler_type: (string) scalar to use ('standard'/'min_max')
        """
        self.X_data = X_data
        self.y_data = y_data
        self.scale = scaler_type

        if scaler_type == "standard":
            scaler = StandardScaler()
        if scaler_type == "min_max":
            scaler = MinMaxScaler()

        self.pca = PCA(n_components=n_components)
        self.pipeline = make_pipeline(scaler, self.pca)


    def transform_data(self):
        """
        Fits-transforms the data through the pipeline

        :return: (X_data, y_data) transformed features, target
        """
        X_data = self.X_data
        y_data = self.y_data

        X_data = self.pipeline.fit_transform(self.X_data)

        return X_data, y_data

    def plot_pca(self, threshold=None, savefig=True):
        """
        Plots pca values, cumulative and individual
        :param threshold: (float range(0,1)) threshold to plot vertical line at
        """
        scaler = self.scale

        # TODO change variable names
        exp_var_pca = self.pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)

        md = ModelPlotter()
        md.plot_pca(exp_var_pca, cum_sum_eigenvalues, scaler, threshold, savefig=True)
