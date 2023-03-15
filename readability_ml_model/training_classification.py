import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

n_cores = 12
k_fold = 5
root = "../"


def training_procedure(model, training_param_grid, verbose=3):
    stratified_k_fold = StratifiedKFold(
        n_splits=k_fold, random_state=seed, shuffle=True
    )
    model_cv = GridSearchCV(
        model,
        param_grid=training_param_grid,
        cv=stratified_k_fold,
        scoring=("neg_mean_squared_error", "r2"),
        refit="neg_mean_squared_error",
        verbose=verbose,
        n_jobs=n_cores,
    ).fit(X_train_normalize, Y_train)
    run_name = type(model_cv.best_estimator_).__name__

    index = np.where(model_cv.cv_results_["rank_test_neg_mean_squared_error"] == 1)[0][
        0
    ]
    with open("results.txt", "a") as file:
        print(run_name, file=file)
        print("Mean test R2:", model_cv.cv_results_["mean_test_r2"][index], file=file)
        print(
            "Mean test neg MSE",
            model_cv.cv_results_["mean_test_neg_mean_squared_error"][index],
            file=file,
        )
        print("Best model specs", model_cv.best_params_, file=file)
        print("\n", file=file)
    print(run_name)
    print(
        "Mean test R2:",
        model_cv.cv_results_["mean_test_r2"][index],
    )
    print(
        "Mean test neg MSE",
        model_cv.cv_results_["mean_test_neg_mean_squared_error"][index],
    )
    print("Best model specs", model_cv.best_params_)
    print("\n")


seed = 42
n_iter = 25000
c_space = 200
alpha_space = 150
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
)

param_grid = {
    "alpha": np.logspace(logspace_low_bound, 0, alpha_space),
    "fit_intercept": [True, False],
}
# Lasso is equivalent to Logistic Regression with l1
training_procedure(
    model=SGDClassifier(
        max_iter=n_iter, random_state=seed, penalty="l1", loss="log_loss"
    ),
    training_param_grid=param_grid,
)

param_grid = {
    "criterion": ["squared_error", "friedman_mse"],
    "max_depth": [32, 64],
}
training_procedure(
    model=DecisionTreeClassifier(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "epsilon": np.logspace(logspace_low_bound, 0, lr_space),
    "C": np.logspace(logspace_low_bound, 0, c_space),
    "fit_intercept": [True, False],
}
training_procedure(model=LinearSVC(random_state=seed), training_param_grid=param_grid)

param_grid = {
    "alpha": np.logspace(logspace_low_bound, 0, alpha_space),
    "fit_intercept": [True, False],
}
# SGD with hinge is a SVM
training_procedure(
    model=SGDClassifier(max_iter=n_iter, random_state=seed, loss="hinge"),
    training_param_grid=param_grid,
)

param_grid = {
    "max_depth": [32, 64],
    "criterion": ["gini", "entropy"],
    "n_estimators": 2 ** np.arange(11)[1:],
}
training_procedure(
    model=RandomForestClassifier(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "learning_rate": np.logspace(logspace_low_bound, 0, lr_space),
    "n_estimators": 2 ** np.arange(11)[1:],
    "loss": ["linear", "square", "exponential"],
}
training_procedure(
    model=AdaBoostClassifier(random_state=seed), training_param_grid=param_grid
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
)
