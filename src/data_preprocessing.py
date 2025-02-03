import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant

def convert_to_numeric(df, columns, mapping):
    for col in columns:
        df[col] = df[col].map(mapping).fillna(0)  # Filling NaN values with 0
    return df

def check_missing_values(data):
    # Finding columns with missing values
    missing_values = data.isnull().sum()

    # Filter out only the columns with missing values
    missing_values = missing_values[missing_values > 0]

    # Calculate the percentage of missing values for each column
    missing_percentage = (missing_values / len(data)) * 100

    # Create a DataFrame to show missing values and their percentages
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})

    # Sort the DataFrame by missing value percentage in descending order
    missing_df = missing_df.sort_values(by='Percentage', ascending=False)

    # Return the result
    return missing_df
    
def misc_val_mapping(x):
    if x == 0:
        return 0
    elif x < 500:
        return 1
    elif x < 1000:
        return 2
    elif x < 5000:
        return 3
    elif x < 10000:
        return 4
    else:
        return 5
        
# Detect outliers using the IQR method
def detect_outliers_iqr(df, features):
    outliers = {}
    for feature in features:
        Q1, Q3 = df[feature].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers[feature] = df[(df[feature] < lower) | (df[feature] > upper)].shape[0]
    return outliers

# Detect outliers using the Z-score method
def detect_outliers_zscore(df, features, threshold=3):
    return {feature: np.sum(np.abs(stats.zscore(df[feature])) > threshold) for feature in features}

# Calculate Variance Inflation Factor (VIF)
def calculate_vif(df, features):
    vif_data = pd.DataFrame({'features': features})
    vif_data['VIF'] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data