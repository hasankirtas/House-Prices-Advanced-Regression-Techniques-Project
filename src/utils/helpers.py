"""
Helper utility functions for data processing and model evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def check_missing_values(data):
    """
    Find columns with missing values and calculate percentages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values count and percentage
    """
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_percentage = (missing_values / len(data)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    missing_df = missing_df.sort_values(by='Percentage', ascending=False)
    
    return missing_df


def remove_outliers_zscore(X, y, threshold=3.0):
    """
    Remove outliers using Z-score method.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series
    threshold : float, default=3.0
        Z-score threshold
    
    Returns
    -------
    X_filtered : pd.DataFrame
        Filtered feature dataframe
    y_filtered : pd.Series
        Filtered target series
    """
    z_scores = zscore(y)
    mask = (z_scores > -threshold) & (z_scores < threshold)
    return X[mask].copy(), y[mask]


def create_neighborhood_encoding(X_train, y_train, alpha=10):
    """
    Create smoothed target encoding for Neighborhood feature.
    Prevents data leakage by using only training data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target (log-transformed)
    alpha : float, default=10
        Smoothing parameter
    
    Returns
    -------
    dict
        Dictionary mapping neighborhood to smoothed average price
    float
        Global mean for fallback
    """
    temp_df = X_train.copy()
    temp_df['SalePrice'] = y_train
    
    neighborhood_stats = temp_df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count'])
    global_mean = y_train.mean()
    
    neighborhood_stats['smoothed'] = (
        neighborhood_stats['mean'] * neighborhood_stats['count'] + global_mean * alpha
    ) / (neighborhood_stats['count'] + alpha)
    
    return neighborhood_stats['smoothed'].to_dict(), global_mean


def apply_neighborhood_encoding(X, neighborhood_encoding, global_mean):
    """
    Apply neighborhood encoding to dataframe.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe
    neighborhood_encoding : dict
        Neighborhood to smoothed price mapping
    global_mean : float
        Global mean for fallback
    
    Returns
    -------
    pd.DataFrame
        Dataframe with Neighborhood_avg_price added and Neighborhood dropped
    """
    X = X.copy()
    X['Neighborhood_avg_price'] = X['Neighborhood'].map(neighborhood_encoding)
    X['Neighborhood_avg_price'].fillna(global_mean, inplace=True)
    X.drop(columns=['Neighborhood'], inplace=True, errors='ignore')
    return X


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns
    -------
    dict
        Dictionary with MSE, RMSE, MAE, and R2
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


def print_metrics(metrics, prefix=""):
    """
    Print metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary with metric values
    prefix : str, default=""
        Prefix to add before each metric line
    """
    print(f"{prefix}MSE: {metrics['MSE']:.2f}")
    print(f"{prefix}RMSE: {metrics['RMSE']:.2f}")
    print(f"{prefix}MAE: {metrics['MAE']:.2f}")
    print(f"{prefix}R2: {metrics['R2']:.4f}")

