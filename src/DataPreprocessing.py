import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

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

class Preprocess:
    """
    Separates columns and target and tunes columns
    """

    def __init__(self, data=None, target="Class"):
        """
        Loads Data and Target to be used
        :param data: path to csv file to be used
        :param target: column name of target
        """

        self.encoder = LabelEncoder()
        if data is None:
            self.df = _sort_class_column(_load_dataset(FILE_NAME))
        else:
            self.df = _sort_class_column(data)

        if target is not None:
            self.target = self.df.pop(target)

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

    def get_features(self):
        """
        Gets df
        :return: X df
        """
        df = self.df

        return df


class FeaturePreprocess:
    """
    Scaling and dimensionality reduction
    """

    def __init__(self, X_data, n_components=15, scaler_type="standard", pca=True):
        """
        Creates a pipeline with the selected scalers and reductors
        :param X_data: (df) feature columns
        :param n_components: (int) number of components to apply pca to
        :param scaler_type: (string) scalar to use ('standard'/'min_max')
        """
        self.X_data = X_data
        self.scale = scaler_type

        if scaler_type == "standard":
            scaler = StandardScaler()
        if scaler_type == "min_max":
            scaler = MinMaxScaler()
        if pca :
            self.pca = PCA(n_components=n_components)
            self.pipeline = make_pipeline(scaler, self.pca)
        if not pca :
            self.pipeline = make_pipeline(scaler)

        self.pipeline.fit(self.X_data)

    def get_scaler_type(self):
        """
        gets the scaler type
        :return: scaler type
        """
        scaler_type = self.scale

        if scaler_type == "standard":
            return "std"
        if scaler_type == "min_max":
            return "min"


    def transform_data(self):
        """
        transforms the data through the pipeline

        :return: (X_data) transformed features
        """
        X_data = self.X_data

        data = self.pipeline.transform(X_data)

        return data

    def transform_prediction(self, data):
        """
        transforms data through the pipeline

        :return: (X_data) transformed features
        """
        data = self.pipeline.transform(data)

        return data

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
        md.plot_pca(exp_var_pca, cum_sum_eigenvalues, scaler, threshold, savefig=savefig)
