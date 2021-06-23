import sys
import os


class Predictor:

    def __init__(self, classifier, pipeline, replace_values=True):
        self.clf = classifier
        self.pipeline = pipeline
        self.replace_values = replace_values

    def predict_proba(self, X_data_raw):
        clf = self.clf
        pipeline = self.pipeline

        return clf.predict_proba(pipeline.transform_prediction(X_data_raw))

    def predict(self, X_data_raw):
        clf = self.clf
        pipeline = self.pipeline

        return clf.predict(pipeline.transform_prediction(X_data_raw))
