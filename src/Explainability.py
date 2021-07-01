import os
from DataPreprocessing import Preprocess
from DataProcessing import ModelTuning, load_file
from ModelExplainability import TabularExplainer
from Main import preprocess_data, get_x_y_set

def load(model_name):
    return load_file(model_name)


if __name__ == '__main__':

    X_data, y_data, y_labels = preprocess_data()

    mt = ModelTuning(X_data, y_data, train_size=0.7, random_state=42)
    X_val, y_val_std = get_x_y_set(mt)
    X_train, y_train = get_x_y_set(mt, type="train")

    # Load both pipelines
    pipeline_std = load("pipeline_std")
    pipeline_min = load("pipeline_min")

    # Load standard models
    rf_std = load("RandomForestClassifier_std")
    ada_svc_std = load("AdaBoostClassifier_SVC_std")
    ada_rf_std = load("AdaBoostClassifier_RandomForestClassifier_std")

    est_std = [rf_std, ada_svc_std, ada_rf_std]

    # Load min_max models
    ada_svc_min = load("AdaBoostClassifier_SVC_min_max")
    rf_min = load("RandomForestClassifier_min_max")
    svc_min = load("SVC_min_max")

    est_min = [ada_svc_min, rf_min, svc_min]

    # Create explainer for standard
    exp = TabularExplainer(est_std, pipeline_std, X_train, y_train, X_val, y_labels)
    exp.explain(clf_index=0, data_index=0)
    exp.explain(clf_index=1, data_index=10)
    exp.explain(clf_index=2, data_index=5)

    # Create explainer for min_max
    exp = TabularExplainer(est_min, pipeline_min, X_train, y_train, X_val, y_labels)
    exp.explain(clf_index=0, data_index=0)
    exp.explain(clf_index=1, data_index=10)
    exp.explain(clf_index=2, data_index=5)
