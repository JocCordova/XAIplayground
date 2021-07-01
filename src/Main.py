import os
import sys
import numpy as np

from DataPreprocessing import Preprocess, FeaturePreprocess
from DataProcessing import ModelTuning, ModelValidating, save_file, load_file

FILE_NAME = os.path.dirname(os.getcwd()) + "\\data" + "\\xAPI-Edu-Data-Edited.csv"

CATEGORICAL_COLUMNS = ["Gender", "Nationality", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic",
                       "Semester", "Relation", "ParentAnsweringSurvey", "ParentSchoolSatisfaction",
                       "StudentAbsenceDays"]

PREFIXES = ["Gender", "Nationality", "PlaceofBirth", "Stage", "Grade", "Section", "Topic",
            "Semester", "Relation", "Survey", "ParentSatisfaction",
            "Absence"]


def preprocess_data(count_missing=False, replace_values=True, encode=True, categorical_columns=CATEGORICAL_COLUMNS,
                    prefixes=PREFIXES):
    """Preprocesses the raw dataset

    Parameters
    ----------
    count_missing : bool, default=False
        Counts all missing values in the dataset
    replace_values : bool, default=True
        Replaces non significative values in the columns "Nationality" and "PlaceofBirth" with "Other"
    encode : bool, default=True
        One Hot encodes categorical columns
    categorical_columns : list of str, defaut=(categorical columns of the dataset)
        Columns to apply one hot encode to
    prefixes : list of str, default="["Gender", "Nationality", "PlaceofBirth", "Stage", "Grade", "Section", "Topic",
            "Semester", "Relation", "Survey", "ParentSatisfaction",
            "Absence"]"
        Prefixes for one hot encoding

    Returns
    ----------
    X_data : pandas df
        feature columns
    y_data : pandas df
        target columns
    y_labels : {ndarray, sparse matrix}
        class labels
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


def preprocess_features(X_data, scaler_type="standard", n_components=None, plot_pca=False, threshold=0.85,
                        savefig=True):
    """Preprocesses feature columns with a scaler and pca

    Parameters
    ----------
    X_data : pandas df
        feature Columns
    scaler_type : str, default="standard"
        scalar to use ('standard'/'min_max')
    n_components : int, default=None
        pca components to use, if 'None' uses all components
    plot_pca : bool, defaut=True
        specifies if pca should be plotted
    threshold : float range(0,1), default=0.85
        pca variance threshold to plot vertical line at
    savefig : bool, default=True
        specifies if pca plot should be saved

    Returns
    ----------
    X_transformed : ndarray
        preprocessed feature columns
    feature_preprocess : feature_preprocess object
        feature_preprocess object used (for the pipeline)
    """
    if n_components is None:
        n_components = len(X_data.columns)

    feature_preprocess = FeaturePreprocess(X_data, n_components=n_components, scaler_type=scaler_type)

    X_transformed = feature_preprocess.transform_data()

    if plot_pca:
        feature_preprocess.plot_pca(threshold=threshold, savefig=savefig)

    return X_transformed, feature_preprocess


def create_estimators(X_data, y_data, train_size=0.7, hyperparam_tune=True, boosting=True, random_state=3,
                      verbose=1):
    """Splits the data in train, test and val, trains three different estimators: Decision Tree, Support Vector Machine
    and Random Forest, can also tune the hyper parameters and boost the estimators with Adaboost

    Parameters
    ----------
    X_data : pandas df
        feature Columns
    y_data : pandas df
        target column
    train_size : float
        Percentage for train
    hyperparam_tune : bool, default=True
        specifies if hyper params should be tuned
    boosting : bool, default=True
        specifies if estimators should be boosted
    random_state : int, default=42
        random state
    verbose : int, default=1
        verbosity level

    Returns
    ----------
    estimators : list of estimators
        trained estimators
    mt : ModelTuning object
        ModelTuning object used (for validation set)
    """
    estimators = []

    mt = ModelTuning(X_data, y_data, train_size, random_state=random_state)

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
    """Gets data set from ModelTuning object

    Parameters
    ----------
    mt : ModelTuning object
        ModelTuning object used
    type : str, default="val"
        specifies which set to return ('train'/'test'/'val')

    Returns
    ----------
    X_data, y_data : ndarray
    """
    if type == "val":
        return mt.get_validation_set()
    if type == "train":
        return mt.get_train_set()
    if type == "test":
        return mt.get_test_set()


def validate_estimators(estimators, X_val, y_val, y_labels, scaler_type="", plot_cf=True, clas_report=True,
                        savefig=True):
    """Validates estimators

    Parameters
    ----------
    estimators : list of estimators
        estimators to validate
    X_val : ndarray
        validation data
    y_val : ndarray
        validation labels
    y_labels : {ndarray, sparse matrix}
        decoded labels
    scaler_type : str, optional
        scaler used ('standard'/'min_max') (for plots)
    plot_cf : bool, default=True
        specifies if confusion matrix should be plot
    clas_report : bool, default=True
        specifies if Classification Report should be printed
    savefig : bool, default=True
        specifies if confusion matrix should be saved as .png
    """
    for est in estimators:

        mv = ModelValidating(est, X_val, y_val, y_labels=y_labels, scaler=scaler_type)
        if plot_cf:
            mv.plot_confusion_matrix(savefig=savefig)
        if clas_report:
            report = mv.classification_report()
            print(f"Classification Report: {est}\n{report}")


def get_n_best(estimators, X_val, y_val, best_n=3, score="f1_score"):
    """Gets best estimators from list

    Parameters
    ----------
    estimators : list of estimators
        list of trained estimators
    X_val : ndarray
        validation data
    y_val : ndarray
        validation labels
    best_n : int, default=3
        number of estimators to pick
    score : str, default="f1_score"
        metric to use for picking best estimators ('accuracy'/'f1_score')

    Returns
    ----------
    best_est : list of estimators of len=´best_n´
    """

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


def save(models, file_name=None, suffix=None):
    """Saves estimator

    Parameters
    ----------
    file_name : str, optional
        name for the file if None model will be saved with suffix
    suffix : str, optional
        suffix to be added
    """
    if file_name is None:
        for model in models:
            save_file(model, suffix=suffix)
    else:
        save_file(models, file_name=file_name)


if __name__ == '__main__':
    print("     Preprocessing...\n")

    X_data, y_data, y_labels = preprocess_data()

    X_data_std, pipeline_std = preprocess_features(X_data, scaler_type="standard", plot_pca=False, n_components=21)
    X_data_min, pipeline_min = preprocess_features(X_data, scaler_type="min_max", plot_pca=False, n_components=13)

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

    save(best_std, suffix="std")
    save(best_min, suffix="min_max")
    save(pipeline_std, file_name="pipeline_std")
    save(pipeline_min, file_name="pipeline_min")
