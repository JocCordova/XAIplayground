import sys
import os


class Predictor:
    """Makes Prediction through the pipeline

    Attributes
    ----------
    clf :  estimator
        estimator to be used
    pipeline : pipeline
        pipeline to be used
    """

    def __init__(self, classifier, pipeline):
        """
        Parameters
        ----------
        classifier : estimator
            estimator to be used
        pipeline : pipeline
            pipeline to be used
        """
        self.clf = classifier
        self.pipeline = pipeline


    def predict_proba(self, X_data_raw):
        """Perform classification with probabilities

        Parameters
        ----------
        X_data_raw : array-like
            raw data to classify

        Returns
        ----------
        ndarray
            probability for each class
        """
        clf = self.clf
        pipeline = self.pipeline

        return clf.predict_proba(pipeline.transform_prediction(X_data_raw))

    def predict(self, X_data_raw):
        """Perform classification

        Parameters
        ----------
        X_data_raw : array-like
            raw data to classify

        Returns
        ----------
        ndarray
            class labels for samples in X_data_raw
        """
        clf = self.clf
        pipeline = self.pipeline

        return clf.predict(pipeline.transform_prediction(X_data_raw))
