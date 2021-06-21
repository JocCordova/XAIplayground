import os
import numpy as np

from DataPreprocessing import Preprocess, FeaturePreprocess
from DataProcessing import ModelTuning, ModelValidating

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"

CATEGORICAL_COLUMNS = ["Gender", "Nationality", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic",
                       "Semester", "Relation", "ParentAnsweringSurvey", "ParentSchoolSatisfaction",
                       "StudentAbsenceDays"]

PREFIXES = ["Gender", "Nationality", "PlaceofBirth", "Stage", "Grade", "Section", "Topic",
            "Semester", "Relation", "Survey", "ParentSatisfaction",
            "Absence"]


def preprocess_data(data=FILE_NAME, count_missing=False, replace_values=True, categorical_columns=CATEGORICAL_COLUMNS,
                    prefixes=PREFIXES):
    """

    :param data:
    :param count_missing:
    :param replace_values:
    :param categorical_columns:
    :param prefixes:
    :return:
    """

    preprocess = Preprocess(data)

    if count_missing:
        print(f"Number of rows missing values: {preprocess.check_missing_values()}")

    if replace_values:
        preprocess.replace_values("Nationality",
                                  ["Lybia", "Iraq", "Lebanon", "Tunisia", "SaudiArabia", "Egypt", "USA", "Venezuela",
                                   "Iran", "Morocco", "Syria", "Palestine"], "Other")
        preprocess.replace_values("PlaceofBirth",
                                  ["Lybia", "Iraq", "Lebanon", "Tunisia", "SaudiArabia", "Egypt", "USA", "Venezuela",
                                   "Iran", "Morocco", "Syria", "Palestine"], "Other")

    preprocess.one_hot_encode(columns=categorical_columns, prefix=prefixes)
    preprocess.target_encode()

    X_data, y_data = preprocess.get_data()
    y_labels = preprocess.target_decode()

    return X_data, y_data, y_labels


def preprocess_features(X_data, scaler_type="standard", n_components=None, plot_pca=False, threshold=0.85, savefig=True):
    """

    :param X_data:
    :param y_data:
    :param n_components:
    :param scaler_type:
    :param plot_pca:
    :param threshold:
    :return:
    """
    if n_components is None:
        n_components = len(X_data.columns)

    feature_preprocess = FeaturePreprocess(X_data, n_components=n_components, scaler_type=scaler_type)

    X_transformed = feature_preprocess.transform_data()

    if plot_pca:
        feature_preprocess.plot_pca(threshold=threshold, savefig=savefig)

    return X_transformed


def create_estimators(X_data, y_data, train=0.7, hyperparam_tune=True, boosting=True, random_state=69,
                      verbose=1):
    """

    :param X_data:
    :param y_data:
    :param test:
    :param val:
    :param hyperparam_tune:
    :param boosting:
    :param random_state:
    :param verbose:
    :return:
    """
    estimators = []

    mt = ModelTuning(X_data, y_data, train, random_state=random_state)

    if verbose > 0:
        print("Creating Basic Estimators...\n")
    dt = mt.create_weak_learner(random_state, verbose, model_type="dt", )
    svm = mt.create_weak_learner(random_state, verbose, model_type="svm")
    rf = mt.create_random_forest(random_state, verbose)

    estimators.extend([dt, svm, rf])

    if hyperparam_tune:
        if verbose > 0:
            print("Tunning Hyperparams...\n")
        tuned_dt = mt.tune_hyperparam(dt, random_state, verbose)
        tuned_svm = mt.tune_hyperparam(svm, random_state, verbose)
        tuned_rf = mt.tune_hyperparam(rf, random_state, verbose)

        estimators.extend([tuned_dt, tuned_svm, tuned_rf])

    if boosting:

        if verbose > 0:
            print("Boosting...\n")
            print("Boosted dt:")
        boosted_dt = mt.boost_weak_learners(tuned_dt, random_state, verbose)

        if verbose > 0:
            print("Boosted svm:")
        boosted_svm = mt.boost_weak_learners(tuned_svm, random_state, verbose)

        if verbose > 0:
            print("Boosted rf:")
        boosted_rf = mt.boost_weak_learners(tuned_rf, random_state, verbose)

        estimators.extend([boosted_dt, boosted_svm, boosted_rf])

    return estimators, mt


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


def validate_estimators(estimators, X_val, y_val, y_labels, scaler_type="", plot_cf=True, clas_report=True,
                        verbose=1, savefig=True):
    for est in estimators:

        mv = ModelValidating(est, X_val, y_val, y_labels=y_labels, scaler=scaler_type)
        if plot_cf:
            mv.plot_confusion_matrix(savefig=savefig)
        if clas_report:
            report = mv.classification_report()
            if verbose > 0:
                print(f"Classification Report: {est}\n{report}")


def get_n_best(estimators, X_val, y_val, best_n=None, score="f1_score"):
    if best_n is None:
        best_n = len(estimators)
        print(best_n)

    best_scores = []

    for est in estimators:
        mv = ModelValidating(est, X_val, y_val, y_labels=y_labels, scaler="")
        indv_scores = mv.get_scores()

        if score == "accuracy":
            best_scores.append(indv_scores[0])
        if score == "f1_score":
            best_scores.append(indv_scores[1])

    best_idx = np.argpartition(best_scores, -best_n)[-best_n:]
    best_est = []
    for index in best_idx:
        best_est.append(estimators[index])

    return best_est


if __name__ == '__main__':
    print("     Preprocessing...\n")

    X_data, y_data, y_labels = preprocess_data()

    X_data_std = preprocess_features(X_data, scaler_type="standard", plot_pca=False, n_components=21)
    X_data_min = preprocess_features(X_data, scaler_type="min_max", plot_pca=False, n_components=13)

    print("     Trainning estimators...\n")

    std_estimators, mt_std = create_estimators(X_data_std, y_data)
    min_estimators, mt_min = create_estimators(X_data_min, y_data)

    print("      Validating estimators...\n")

    X_val_std, y_val_std = get_x_y_set(mt_std)
    X_val_min, y_val_min = get_x_y_set(mt_min)

    best_std = get_n_best(std_estimators, X_val_std, y_val_std, best_n=3)
    print("     Best Standard_scaled est:", best_std)

    best_min = get_n_best(min_estimators, X_val_min, y_val_min, best_n=3)
    print("     Best Min_´Max_scaled est:", best_min)

    validate_estimators(best_std, X_val_std, y_val_std, y_labels, scaler_type="standard")
    validate_estimators(best_min, X_val_min, y_val_min, y_labels, scaler_type="min_max")
