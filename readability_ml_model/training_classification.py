import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    SGDClassifier,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

n_cores_1 = 12
n_cores_2 = 6
k_fold = 5
root = "../"


def training_procedure(model, training_param_grid, verbose=3, n_cores=4):
    stratified_k_fold = StratifiedKFold(
        n_splits=k_fold, random_state=seed, shuffle=True
    )
    accuracy = make_scorer(accuracy_score)
    precision = make_scorer(precision_score, average="macro", zero_division=0)
    recall = make_scorer(recall_score, average="macro", zero_division=0)
    f1 = make_scorer(f1_score, average="macro", zero_division=0)

    model_cv = GridSearchCV(
        model,
        param_grid=training_param_grid,
        cv=stratified_k_fold,
        scoring={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        refit="f1",
        verbose=verbose,
        n_jobs=n_cores,
    ).fit(X_train_normalize, Y_train)

    Y_pred = model_cv.best_estimator_.predict(X_test_normalize)
    test_f1_score = f1_score(
        y_true=Y_test, y_pred=Y_pred, average="macro", zero_division=0
    )
    test_accuracy = accuracy_score(y_true=Y_test, y_pred=Y_pred)
    test_precision = precision_score(
        y_true=Y_test, y_pred=Y_pred, average="macro", zero_division=0
    )
    test_recall = recall_score(
        y_true=Y_test, y_pred=Y_pred, average="macro", zero_division=0
    )

    run_name = type(model_cv.best_estimator_).__name__
    best_model_specs = model_cv.best_params_

    with open("results_classification.txt", "a") as file:
        print(run_name, file=file)
        print("Mean test F1:", test_f1_score, file=file)
        print("Mean test accuracy:", test_accuracy, file=file)
        print("Mean test precision:", test_precision, file=file)
        print("Mean test recall:", test_recall, file=file)
        print("Best model specs:", best_model_specs, file=file)
        print("\n", file=file)

    print(run_name)
    print("Mean test F1:", test_f1_score)
    print("Mean test accuracy:", test_accuracy)
    print("Mean test precision:", test_precision)
    print("Mean test recall:", test_recall)
    print("Best model specs:", best_model_specs)
    print("\n")


seed = 42
n_iter = 25000
c_space = 100
alpha_space = 100
lr_space = 100
logspace_low_bound = -6

all_data = pd.read_csv(os.path.join(root, "datastore", "pre_process_newsela_data.csv"))

X, Y = (all_data.loc[:, "1persProns":], all_data["Y_class"])

# We split dataset into 80-20 train-test sets.
# Later for training, we will use a k-fold for the train-val sets.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=seed, shuffle=True, stratify=Y
)

standard_scaler = StandardScaler()
X_train_normalize = standard_scaler.fit_transform(X_train)
X_test_normalize = standard_scaler.transform(X_test)

print("Training procedure")

param_grid = {
    "alpha": np.logspace(logspace_low_bound, 0, alpha_space),
    "fit_intercept": [True, False],
}
# Logistic Regression is equivalent to SGD with a log loss
training_procedure(
    model=SGDClassifier(max_iter=n_iter, random_state=seed, loss="log_loss"),
    training_param_grid=param_grid,
    n_cores=n_cores_1,
)

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [32, 64],
}
training_procedure(
    model=DecisionTreeClassifier(random_state=seed),
    training_param_grid=param_grid,
    n_cores=n_cores_1,
)

param_grid = {
    "alpha": np.logspace(logspace_low_bound, 0, alpha_space),
    "fit_intercept": [True, False],
}
# SGD with hinge is a SVM
training_procedure(
    model=SGDClassifier(max_iter=n_iter, random_state=seed, loss="hinge"),
    training_param_grid=param_grid,
    n_cores=n_cores_2,
)

param_grid = {
    "max_depth": [32, 64],
    "criterion": ["gini", "entropy"],
    "n_estimators": 2 ** np.arange(11)[1:],
}
training_procedure(
    model=RandomForestClassifier(random_state=seed),
    training_param_grid=param_grid,
    n_cores=n_cores_2,
)

param_grid = {
    "learning_rate": np.logspace(logspace_low_bound, 0, lr_space),
    "n_estimators": 2 ** np.arange(11)[1:],
    "loss": ["linear", "square", "exponential"],
}
training_procedure(
    model=AdaBoostClassifier(random_state=seed),
    training_param_grid=param_grid,
    n_cores=n_cores_2,
)

param_grid = {
    "hidden_layer_sizes": [
        x
        for i in range(3, 5)
        for x in itertools.product(
            (
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
            ),
            repeat=i,
        )
    ],
}
training_procedure(
    model=MLPClassifier(
        random_state=seed, max_iter=n_iter, n_iter_no_change=25, solver="adam"
    ),
    training_param_grid=param_grid,
    n_cores=n_cores_2,
)
