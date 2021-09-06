import pandas as pd
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

from DataExploration import ModelPlotter

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"
BASIC_DECODER = [1, 2, 0]


def _load_dataset(csv_file):
    # Loads a csv file as a data frame and returns it

    return pd.read_csv(csv_file)


def _sort_class_column(df):
    # Sorts the dataframe by the "Class" column

    categories = ["L", "M", "H"]

    df["Class"] = pd.Categorical(df["Class"], categories=categories)
    df.sort_values(by="Class")

    return df


class Preprocess:
    """Separates columns and target and tunes columns

    Attributes
    ----------
    encoder : encoder
        label encoder
    df : pandas df
        feature columns
    target : str, default="Class"
        column name of target
    """

    def __init__(self, data=None, target="Class"):
        """Loads Data and Target to be used

        Parameters
        ----------
        data : str, optional
            path to csv file to be used
        target : str, default="Class"
            column name of target
        """

        self.encoder = LabelEncoder()
        if data is None:
            self.df = _sort_class_column(_load_dataset(FILE_NAME))
        else:
            self.df = _load_dataset(data)

        self.target = target

    def check_missing_values(self):
        """Counts all missing row values

        Returns
        ----------
        int
            sum of missing values
        """

        df = self.df

        null_cols = df.isnull().any(axis=1).sum()

        return null_cols

    def replace_values(self, column, old_values, new_value):
        """Replaces categorical values in a specific column

        Parameters
        ----------
        column : str
            column where the value is
        old_values : str or list of str
            value to be replaced
        new_value : str or list of str
            value to be replaced to
        """

        for old_value in old_values:
            self.df.loc[(self.df[column] == old_value), column] = new_value

    def remove_values(self, column, values):
        """Removes row in dataframe if the values exist in a specific column

        Parameters
        ----------
        column : str
            column where the value is
        values : str or list of str
            value to be removed
        """

        for value in values:
            self.df = self.df[self.df[column] != value]

    def one_hot_encode(self, columns, prefix):
        """One Hot Encodes values in a specific column with a specific prefix

         Parameters
         ----------
         columns : str or list of str
             columns to be encoded
         prefix : str or list of str
             prefix to be used
        """

        self.df = pd.get_dummies(self.df, columns=columns, prefix=prefix)

    def target_encode(self):
        """Encodes target column
        """

        self.target = self.df.pop(self.target)
        self.target = self.encoder.fit_transform(self.target)

    def target_decode(self, target=BASIC_DECODER):
        """Decodes list using using target decoder

        Parameters
        ----------
        target : {array-like, sparse matrix} , defaut=[0, 1, 2]
            target to decode

        Returns
        ----------
        {ndarray, sparse matrix}
            decoded target
        """

        return self.encoder.inverse_transform(target)

    def get_data(self):
        """Gets df and target
        """
        df = self.df
        target = self.target

        return df, target

    def get_features(self):
        """Gets df
        """
        df = self.df

        return df


class FeaturePreprocess:
    """Scaling and dimensionality reduction

    Attributes
    ----------
    X_data : pandas df
        feature columns
    scale : str
        scalar used
    pca : pca
        pca used
    pipeline : pipeline
        pipeline used
    """

    def __init__(self, X_data, n_components=15, scaler_type="standard", pca=True):
        """Creates a pipeline with the selected scalers and

        Parameters
        ----------
        X_data : pandas df
            feature columns
        n_components : int
            number of components to apply pca to
        scaler_type : str
            scalar to use ('standard'/'min_max')
        pca : bool, default=True
            specifies if pca should be applied
        """

        self.X_data = X_data
        self.scale = scaler_type

        if scaler_type == "standard":
            scaler = StandardScaler()
        if scaler_type == "min_max":
            scaler = MinMaxScaler()
        if pca:
            self.pca = PCA(n_components=n_components)
            self.pipeline = make_pipeline(scaler, self.pca)
        if not pca:
            self.pipeline = make_pipeline(scaler)

        self.pipeline.fit(self.X_data)

    def get_scaler_type(self):
        """gets the scaler type
        """
        scaler_type = self.scale

        if scaler_type == "standard":
            return "std"
        if scaler_type == "min_max":
            return "min"

    def transform_data(self):
        """transforms the data through the pipeline

        Returns
        ----------
        {ndarray, sparse matrix}
            transformed features
        """
        X_data = self.X_data

        data = self.pipeline.transform(X_data)

        return data

    def transform_prediction(self, data):
        """transforms prediction through the pipeline

        Parameters
        ----------
        data : pandas df
            data to be tranformed

        Returns
        ----------
        {ndarray, sparse matrix}
            transformed features

        """
        data = self.pipeline.transform(data)

        return data

    def plot_pca(self, threshold=None, savefig=True):
        """Plots pca values, cumulative and individual

        Parameters
        ----------
        threshold : float range(0,1)
            threshold to plot vertical line at
        savefig : bool, default=True
            specifies if plot should be saved as .png
        """

        scaler = self.scale

        var_pca = self.pca.explained_variance_ratio_
        sum_eigenvalues = np.cumsum(var_pca)

        md = ModelPlotter()
        md.plot_pca(var_pca, sum_eigenvalues, scaler, threshold, savefig=savefig)
