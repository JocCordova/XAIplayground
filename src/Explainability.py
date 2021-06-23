import os
import sys
import numpy as np
from DataPreprocessing import Preprocess
from DataProcessing import ModelTuning, load_file
from ModelExplainability import TabularExplainer

sys.path.insert(1, os.path.dirname(os.getcwd()) + "\\pipeline")
from DataPipeline import preprocess_data

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"

CATEGORICAL_COLUMNS = ["Gender", "Nationality", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic",
                       "Semester", "Relation", "ParentAnsweringSurvey", "ParentSchoolSatisfaction",
                       "StudentAbsenceDays"]

PREFIXES = ["Gender", "Nationality", "PlaceofBirth", "Stage", "Grade", "Section", "Topic",
            "Semester", "Relation", "Survey", "ParentSatisfaction",
            "Absence"]


def preprocess_data(count_missing=False, replace_values=True, encode=True, categorical_columns=CATEGORICAL_COLUMNS,
                    prefixes=PREFIXES):
    """

    :param encode:
    :param data:
    :param count_missing:
    :param replace_values:
    :param categorical_columns:
    :param prefixes:
    :return:
    """

    preprocess = Preprocess()

    if count_missing:
        print(f"Number of rows missing values: {preprocess.check_missing_values()}")

    if replace_values:
        preprocess.replace_values("Nationality",
                                  ["Lybia", "Iraq", "Lebanon", "Tunisia", "SaudiArabia", "Egypt", "USA", "Venezuela",
                                   "Iran", "Morocco", "Syria", "Palestine"], "Other")
        preprocess.replace_values("PlaceofBirth",
                                  ["Lybia", "Iraq", "Lebanon", "Tunisia", "SaudiArabia", "Egypt", "USA", "Venezuela",
                                   "Iran", "Morocco", "Syria", "Palestine"], "Other")
    if encode:
        preprocess.one_hot_encode(columns=categorical_columns, prefix=prefixes)
    preprocess.target_encode()
    X_data, y_data = preprocess.get_data()
    y_labels = preprocess.target_decode()

    return X_data, y_data, y_labels


def get_x_y_set(mt, type="val"):
    """

    :param mt:
    :param type:
    :return:
    """
    if type == "val":
        return mt.get_validation_set()
    if type == "train":
        return mt.get_train_set()
    if type == "test":
        return mt.get_test_set()


def load(model_name):
    return load_file(model_name)


if __name__ == '__main__':
    X_data, y_data, y_labels = preprocess_data()

    mt = ModelTuning(X_data, y_data, train=0.7, random_state=42)
    X_val, y_val_std = get_x_y_set(mt)
    X_train, y_train = get_x_y_set(mt, type="train")


    ada_std = load("AdaBoostClassifier_SVC_std")
    ada_min = load("RandomForestClassifier_min_max")
    pipeline_std = load("pipeline_std")
    pipeline_min = load("pipeline_min")

    est = [ada_std, ada_min]
    exp = TabularExplainer(est, pipeline_std, X_train, y_train, X_val, y_labels)
    exp.explain(clf_index=0, data_index=54)
    exp.explain(clf_index=0, data_index=25)
    exp.explain(clf_index=0, data_index=42)
    exp = TabularExplainer(est, pipeline_min, X_train, y_train, X_val, y_labels)
    exp.explain(clf_index=1, data_index=3)
    exp.explain(clf_index=1, data_index=15)
    exp.explain(clf_index=1, data_index=36)
