import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from zca import ZCA

n_cores = 12
k_fold = 5
root = "../"
whitening = False


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

    Y_pred = model_cv.best_estimator_.predict(X_test_normalize)
    test_r2 = r2_score(y_true=Y_test, y_pred=Y_pred)
    test_mean_new_mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)

    run_name = type(model_cv.best_estimator_).__name__
    best_model_specs = model_cv.best_params_

    with open("results_regression.txt", "a") as file:
        print(run_name, file=file)
        print("Mean test R2:", test_r2, file=file)
        print("Mean test neg MSE:", test_mean_new_mse, file=file)
        print("Best model specs:", best_model_specs, file=file)
        print("\n", file=file)

    print(run_name)
    print("Mean test R2:", test_r2)
    print("Mean test neg MSE:", test_mean_new_mse)
    print("Best model specs:", best_model_specs)
    print("\n")


seed = 42
n_iter = 25000
c_space = 100
alpha_space = 100
lr_space = 100
logspace_low_bound = -6

all_data = pd.read_csv(os.path.join(root, "datastore", "pre_process_newsela_data.csv"))

X, Y = (all_data.loc[:, "1persProns":], all_data["Y"])

# We split dataset into 80-20 train-test sets.
# Later for training, we will use a k-fold for the train-val sets.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=seed, shuffle=True, stratify=Y
)

if whitening:
    zca_scaler = ZCA()
    X_train_normalize = zca_scaler.fit_transform(X_train)
    X_test_normalize = zca_scaler.transform(X_test)
else:
    standard_scaler = StandardScaler()
    X_train_normalize = standard_scaler.fit_transform(X_train)
    X_test_normalize = standard_scaler.transform(X_test)

print("---Training procedure---")

param_grid = {
    "fit_intercept": [True, False],
}
training_procedure(model=LinearRegression(), training_param_grid=param_grid)

param_grid = {
    "alpha": np.logspace(logspace_low_bound, 0, alpha_space),
    "fit_intercept": [True, False],
}
training_procedure(
    model=Lasso(max_iter=n_iter, random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "criterion": ["squared_error", "friedman_mse"],
    "max_depth": [32, 64],
}
training_procedure(
    model=DecisionTreeRegressor(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "epsilon": np.logspace(logspace_low_bound, 0, lr_space),
    "C": np.logspace(logspace_low_bound, 0, c_space),
    "fit_intercept": [True, False],
}
training_procedure(model=LinearSVR(random_state=seed), training_param_grid=param_grid)

param_grid = {
    "max_depth": [32, 64],
    "criterion": ["squared_error", "friedman_mse"],
    "n_estimators": 2 ** np.arange(11)[1:],
}
training_procedure(
    model=RandomForestRegressor(random_state=seed), training_param_grid=param_grid
)

param_grid = {
    "learning_rate": np.logspace(logspace_low_bound, 0, lr_space),
    "n_estimators": 2 ** np.arange(11)[1:],
    "loss": ["linear", "square", "exponential"],
}
training_procedure(
    model=AdaBoostRegressor(random_state=seed), training_param_grid=param_grid
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
    model=MLPRegressor(
        random_state=seed, max_iter=n_iter, n_iter_no_change=25, solver="adam"
    ),
    training_param_grid=param_grid,
)
