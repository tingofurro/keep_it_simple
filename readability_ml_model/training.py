import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    BayesianRidge,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

n_cores = 4
root = "../"


def training_procedure(model, training_param_grid, verbose=1):
    stratified_k_fold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
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
n_iter = 10000
c_space = 500

all_data = pd.read_csv(os.path.join(root, "datastore", "pre_process_newsela_data.csv"))

X, Y = (all_data.loc[:, all_data.columns != "Y"], all_data["Y"])

# We split dataset into 80-20 train-test sets.
# Later for training, we will use a 10-fold for the train-val sets.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=seed, shuffle=True, stratify=Y
)

standard_scaler = StandardScaler()
X_train_normalize = standard_scaler.fit_transform(X_train)
X_test_normalize = standard_scaler.transform(X_test)

print("Training procedure")

param_grid = {
    "fit_intercept": [True, False],
}
training_procedure(
    model=LinearRegression(n_jobs=n_cores), training_param_grid=param_grid
)

param_grid = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": np.logspace(-14, 0, c_space),
    # "tol": np.logspace(-16, -6, 10),
    "fit_intercept": [True, False],
}
training_procedure(
    model=LogisticRegression(n_jobs=n_cores, max_iter=n_iter, random_state=seed),
    training_param_grid=param_grid,
)

param_grid = {
    "alpha": np.logspace(-14, 0, 250),
    # "tol": np.logspace(-16, -6, 10),
    "fit_intercept": [True, False],
}
training_procedure(
    model=Ridge(max_iter=n_iter, random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "alpha": np.logspace(-14, 0, 250),
    # "tol": np.logspace(-16, -6, 10),
    "fit_intercept": [True, False],
}
training_procedure(
    model=Lasso(max_iter=n_iter, random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "alpha_1": np.logspace(-14, 0, 10),
    "alpha_2": np.logspace(-14, 0, 10),
    "lambda_1": np.logspace(-14, 0, 10),
    "lambda_2": np.logspace(-14, 0, 10),
    # "tol": np.logspace(-16, -6, 10),
    "fit_intercept": [True, False],
}
training_procedure(model=BayesianRidge(n_iter=n_iter), training_param_grid=param_grid)

param_grid = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "max_depth": [32, 64, 128],
}
training_procedure(
    model=DecisionTreeRegressor(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "epsilon": np.logspace(-14, 0, 250),
    # "tol": np.logspace(-16, -6, 10),
    "C": np.logspace(-14, 0, c_space),
    "fit_intercept": [True, False],
}
training_procedure(model=LinearSVR(random_state=seed), training_param_grid=param_grid)

param_grid = {
    "max_depth": [32, 64, 128],
    "criterion": ["gini", "entropy"],
    "n_estimators": np.arange(2, 4096, step=2),
}
training_procedure(
    model=RandomForestRegressor(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "hidden_layer_sizes": [
        x
        for i in range(3, 6)
        for x in itertools.product(
            (
                2,
                5,
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
                80,
                85,
                90,
                95,
                100,
                105,
                110,
                115,
                120,
            ),
            repeat=i,
        )
    ],
    "tol": np.logspace(-14, -1, 10),
    "activation": ["logistic", "relu"],
}
training_procedure(
    model=MLPRegressor(
        random_state=seed, max_iter=n_iter, n_iter_no_change=25, solver="adam"
    ),
    training_param_grid=param_grid,
)
