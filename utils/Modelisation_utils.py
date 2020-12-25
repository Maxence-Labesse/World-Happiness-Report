import pandas as pd
import numpy as np
import random

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score




def train_test_metrics(y_test, y_pred_test, y_train=[], y_pred_train=[], display=True):
    """
    compute MSE for train and test target labels.
    
    Parameters
    ----------
    y_test: list
        target test real values
        
    y_pred_test: list
        target test predicted values
        
    y_train: list (default: [])
        target train real values
        
    y_pred_train: list (default:[])
        target train predicted values
        
    display: boolean (default:False)
        if True, print MSE values for train and test
    """
    d_mse = {"test": None,"train": None}
    d_mse["test"] = mean_squared_error(y_test, y_pred_test)
    
    if display:
        print("test_set mse:", d_mse["test"])
    
    if len(y_train)>0:
        d_mse["train"] = mean_squared_error(y_train, y_pred_train)
        if display:
            print("train_set mse:", d_mse["train"])

    return d_mse

####################################################################


def compare_algo(df_train, df_test, target, l_features, y_test_benchmark=[]):
    """
    Compare different Machine Learning Alrogithms for regression
    
    df_train: DataFrame
        train set
        
    df_test: DataFrame
        test set
        
    target: string
        target variable name
        
    l_features: list
        predictors list
        
    y_test_benchmark: list (default:[])
        predicted target test values with benchmark model
    """
    X_train = df_train[l_features].copy()
    y_train = df_train[target].copy()

    X_test = df_test[l_features].copy()
    y_test = df_test[target].copy()
    
    # Benchmark
    if len(y_test_benchmark)>0:
        benchmark_mse = train_test_metrics(y_test=y_test, y_pred_test=y_test_benchmark, 
                                            display=False)
    
    # Linear regression
    model = LinearRegression() 
    linear_fit = model.fit(X_train, y_train)
    y_pred_lin = linear_fit.predict(X_test)
    y_train_pred_lin = linear_fit.predict(X_train)
    
    lreg_mse = train_test_metrics(y_test, y_pred_lin, y_train, y_train_pred_lin, 
                                                       display=False)
    
    # Polynomial regression
    poly_reg = PolynomialFeatures(degree = 2)
    X_train_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.transform(X_test)
    
    lin_reg2 = LinearRegression()
    lin_reg2.fit(X_train_poly,y_train)
    y_train_pred_reg2 = lin_reg2.predict(X_train_poly)
    y_test_pred_reg2 = lin_reg2.predict(X_test_poly)

    reg2_mse = train_test_metrics(y_test, y_test_pred_reg2, y_train, y_train_pred_reg2, 
                                                       display=False)
    
    # Random Forest
    l_n_estimators= np.random.uniform(low=20, high=500, size=40).astype(int)
    l_max_features= ['auto', 'log2']
    l_max_depth= [3, 4, 5, 6, 7, 8]
    l_min_samples_split= [5, 10, 15, 20]
    
    for i in range(40):
        n_estimators = random.choice(l_n_estimators)
        max_features = random.choice(l_max_features)
        max_depth = random.choice(l_max_depth)
        min_samples_split = random.choice(l_min_samples_split)

        RF_reg = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,
                                       max_depth=max_depth, min_samples_split=min_samples_split)

        RF_reg.fit(X_train, y_train)
        y_train_pred_rf = RF_reg.predict(X_train)
        y_test_pred_rf = RF_reg.predict(X_test)
    
        rf_mse = train_test_metrics(y_test, y_test_pred_rf, y_train, y_train_pred_rf, 
                                                       display=False)
    
        best_valid_model_mse = 1000
        if (abs(rf_mse['test']-rf_mse['train'])<2 and rf_mse['test']<best_valid_model_mse) or i==1:
            best_valid_model_mse = rf_mse['test']
            best_valid_model_mse_train = rf_mse['train']
            best_model_n_estimators = n_estimators 
            best_model_max_features = max_features
            best_model_max_depth = max_depth
            best_model_min_samples_split = min_samples_split
    
    
    # Display
    if len(y_test_benchmark)>0:
        l_idx = ["Benchmark", "Linear Regression", "Polynomial (square) Regression", "Random Forest"]
    else:
        l_idx = ["Linear Regression", "Polynomial (square) Regression", "Random Forest"]
    
    df_mse = pd.DataFrame(index=l_idx, columns=['Test MSE', "Train MSE"])
    
    
    if len(y_test_benchmark)>0:
        df_mse.loc["Benchmark"] = list(benchmark_mse.values())
    df_mse.loc["Linear Regression"] = list(lreg_mse.values())
    df_mse.loc["Polynomial (square) Regression"] = list(reg2_mse.values())
    df_mse.loc["Random Forest"] = list(rf_mse.values())
    
    print("RF params: \nn_estimators:",best_model_n_estimators,
          "/ max_features:",best_model_max_features,
          "/ max_depth:",best_model_max_depth,
          "/ min_samples_split",best_model_min_samples_split)
    
    display(df_mse)
    

    
#    return y_pred_lin, y_test