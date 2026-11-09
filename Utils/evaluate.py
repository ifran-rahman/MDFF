import math
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

models_dir = 'models'


def save_model(model, filename):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    joblib.dump(model, path)
    print(f"Model saved as {path}")


def report(name, y_true, predictions):
    mse = mean_squared_error(y_true, predictions)
    print(f'************ {name} ************')
    print("MSE:", mse)
    print("RMSE:", math.sqrt(mse))
    print("MAE:", mean_absolute_error(y_true, predictions))
    print("R2 Score:", r2_score(y_true, predictions))


def _fit_and_report(model, name, filename, train, val):
    """Last column of `train`/`val` is the target; the rest are PCA features."""
    model.fit(train[:, :-1], train[:, -1])
    save_model(model, filename)
    report(name, val[:, -1], model.predict(val[:, :-1]))


def train_lr_model(train, val):
    _fit_and_report(LinearRegression(), "Linear Regression", "regression_model.pkl", train, val)


def train_decision_tree_model(train, val):
    _fit_and_report(DecisionTreeRegressor(random_state=42), "Decision Tree Regression", "decision_tree_model.pkl", train, val)


def train_random_forest_model(train, val):
    _fit_and_report(RandomForestRegressor(random_state=42, n_jobs=4), "Random Forest Regression", "random_forest_model.pkl", train, val)


def train_svm_model(train, val):
    _fit_and_report(LinearSVR(dual="auto", random_state=42), "Support Vector Machine", "svr_model.pkl", train, val)


def _chunked_least_squares(X, y, transform, chunk):
    """Ordinary least squares on transform(X) without materialising it.

    Uses a streaming QR (TSQR) update, so peak memory is one chunk instead of the
    full design matrix (~5 GB for degree-3 features on the full dataset). Gives
    the same solution as LinearRegression on the full matrix.
    """
    R = None
    for i in range(0, len(X), chunk):
        A = np.hstack((transform(X[i:i + chunk]), y[i:i + chunk, None]))
        R = np.linalg.qr(A if R is None else np.vstack((R, A)), mode='r')
    # Least squares on the small triangular factor; also handles fewer rows than features.
    return np.linalg.lstsq(R[:, :-1], R[:, -1], rcond=None)[0]


def train_poly_model(train, val, degree, chunk=20000):
    """Polynomial features + linear regression. Saved as (model, transformer)."""
    poly_features = PolynomialFeatures(degree=degree).fit(train[:1, :-1])
    coef = _chunked_least_squares(train[:, :-1], train[:, -1], poly_features.transform, chunk)

    # Wrap as a fitted LinearRegression; the bias column of PolynomialFeatures carries the intercept.
    model = LinearRegression()
    model.coef_, model.intercept_, model.n_features_in_ = coef, 0.0, len(coef)
    save_model((model, poly_features), f"polynomial_regression_model_degree_{degree}.pkl")

    X_val = val[:, :-1]
    predictions = np.concatenate([model.predict(poly_features.transform(X_val[i:i + chunk]))
                                  for i in range(0, len(X_val), chunk)])
    report(f"Polynomial Regression (degree {degree})", val[:, -1], predictions)
