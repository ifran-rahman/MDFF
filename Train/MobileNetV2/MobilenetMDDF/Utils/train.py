from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error 
from sklearn.svm import SVR
import pickle
import os

os.mkdir('Models') if not os.path.exists('Models') else None

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math


## Load PCA
def train_lr_model(reducted_train_features, reducted_val_features):
  # Initialize the linear regression model
  regression_model = LinearRegression()

  # Train the model
  regression_model.fit(reducted_train_features[:, :-1], reducted_train_features[:, -1])

  # Make predictions
  predictions = regression_model.predict(reducted_val_features[:, :-1])

  print('************ Linear Regression ************')
  mse = mean_squared_error(reducted_val_features[:, -1], predictions)
  rmse = math.sqrt(mse)
  mae = mean_absolute_error(reducted_val_features[:, -1], predictions)
  print("MSE:", mse)
  print("RMSE:", rmse)
  print("MAE:", mae)


def train_decision_tree_model(reducted_train_features, reducted_val_features):
    # Initialize the decision tree regression model
    decision_tree_model = DecisionTreeRegressor()

    # Train the model
    decision_tree_model.fit(reducted_train_features[:, :-1], reducted_train_features[:, -1])

    # Make predictions
    predictions = decision_tree_model.predict(reducted_val_features[:, :-1])

    print('************ Decision Tree Regression ************')
    mse = mean_squared_error(reducted_val_features[:, -1], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(reducted_val_features[:, -1], predictions)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)


def train_random_forest_model(reducted_train_features, reducted_val_features):
    # Initialize the random forest regression model
    random_forest_model = RandomForestRegressor()

    # Train the model
    random_forest_model.fit(reducted_train_features[:, :-1], reducted_train_features[:, -1])

    # Make predictions
    predictions = random_forest_model.predict(reducted_val_features[:, :-1])

    print('************ Random Forest Regression ************')
    mse = mean_squared_error(reducted_val_features[:, -1], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(reducted_val_features[:, -1], predictions)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)


def train_svm_model(reducted_train_features, reducted_val_features):
    
    svr_model = SVR()
    svr_model.fit(reducted_train_features[:, :-1], reducted_train_features[:, -1])
    predictions = svr_model.predict(reducted_val_features[:, :-1])

    print('************ Support Vector Machine ************')
    mse = mean_squared_error(reducted_val_features[:, -1], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(reducted_val_features[:, -1], predictions)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)


from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures

def train_svm_model(reducted_train_features, reducted_val_features):
    
    svr_model = LinearSVR(dual='auto')
    svr_model.fit(reducted_train_features[:, :-1], reducted_train_features[:, -1])
    predictions = svr_model.predict(reducted_val_features[:, :-1])

    print('************ Support Vector Machine ************')
    mse = mean_squared_error(reducted_val_features[:, -1], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(reducted_val_features[:, -1], predictions)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)

def train_poly_model(reducted_train_features, reducted_val_features, degree):
    regression_model = LinearRegression()  # You can choose the appropriate alpha value
    
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(reducted_train_features[:, :-1])
    pickle.dump(poly_features, open('poly_features_mobilenet.sav', 'wb'))
    
    regression_model.fit(X_train_poly, reducted_train_features[:, -1])
    
    X_val_poly = poly_features.transform(reducted_val_features[:, :-1])
    pickle.dump(regression_model, open('poly_regression_mobilenet.sav', 'wb'))
    predictions = regression_model.predict(X_val_poly)

    print('************ Polynomial Regression ************')
    mse = mean_squared_error(reducted_val_features[:, -1], predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(reducted_val_features[:, -1], predictions)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)

