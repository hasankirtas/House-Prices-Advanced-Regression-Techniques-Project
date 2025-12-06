"""
Feature engineering transformations.
Creates new features from existing ones to improve model performance.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Create new features from existing ones."""
    
    def __init__(self, include_bsmt_quality_interaction=True):
        """
        Parameters
        ----------
        include_bsmt_quality_interaction : bool, default=True
            Whether to include interaction term between TotalFinishedBsmtSF and BsmtQual
        """
        self.include_bsmt_quality_interaction = include_bsmt_quality_interaction
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Create new engineered features."""
        df = X.copy()
        
        # Calculate total finished basement area
        if 'BsmtFinSF1' in df.columns and 'BsmtFinSF2' in df.columns:
            df['TotalFinishedBsmtSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
            
            # Add interaction term if enabled and BsmtQual is numeric
            if (self.include_bsmt_quality_interaction and 
                'BsmtQual' in df.columns and 
                pd.api.types.is_numeric_dtype(df['BsmtQual'])):
                df['TotalFinishedBsmtSF_BsmtQual_Interaction'] = (
                    df['TotalFinishedBsmtSF'] * df['BsmtQual']
                )
        
        # Combine full and half bathrooms counts (basement + above ground)
        if 'BsmtFullBath' in df.columns and 'FullBath' in df.columns:
            df['TotalFullBaths'] = df['BsmtFullBath'] + df['FullBath']
        
        if 'BsmtHalfBath' in df.columns and 'HalfBath' in df.columns:
            df['TotalHalfBaths'] = df['BsmtHalfBath'] + df['HalfBath']
        
        # House age and years since last remodel (no negative values)
        if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
        if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
            df['YearsSinceRemodel'] = (
                (df['YrSold'] - df['YearRemodAdd']).apply(lambda x: x if x >= 0 else 0)
            )
        
        # Sum all porch area types into one feature
        porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        existing_porch_cols = [col for col in porch_cols if col in df.columns]
        if existing_porch_cols:
            df['TotalPorchArea'] = df[existing_porch_cols].sum(axis=1)
        
        return df

