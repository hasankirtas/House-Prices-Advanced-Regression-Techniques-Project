"""
Custom transformers for data preprocessing.
All transformers follow scikit-learn's BaseEstimator and TransformerMixin pattern.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FillLotFrontageByNeighborhood(BaseEstimator, TransformerMixin):
    """Fill missing LotFrontage values with median for each neighborhood."""
    
    def fit(self, X, y=None):
        """Learn neighborhood medians from training data."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self.neigh_medians_ = X.groupby('Neighborhood')['LotFrontage'].median()
        return self
    
    def transform(self, X):
        """Apply learned medians to fill missing values."""
        X = X.copy()
        
        def fill_func(row):
            if pd.isna(row['LotFrontage']):
                return self.neigh_medians_.get(row['Neighborhood'], np.nan)
            else:
                return row['LotFrontage']
        
        X['LotFrontage'] = X.apply(fill_func, axis=1)
        return X


class FillGarageCols(BaseEstimator, TransformerMixin):
    """Fill garage-related columns with 'None' and GarageYrBlt with 0."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_cols:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        if 'GarageYrBlt' in X.columns:
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
        return X


class FillBsmtCols(BaseEstimator, TransformerMixin):
    """Fill basement-related columns with 'None'."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in bsmt_cols:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        return X


class FillNoneCols(BaseEstimator, TransformerMixin):
    """Fill specified columns with 'None'."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        return X


class FillMasVnrType(BaseEstimator, TransformerMixin):
    """Fill MasVnrType with most frequent value."""
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'MasVnrType' in X.columns:
            mode_result = X['MasVnrType'].mode()
            self.most_frequent_ = mode_result[0] if len(mode_result) > 0 else 'None'
        else:
            self.most_frequent_ = 'None'
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'MasVnrType' in X.columns:
            X['MasVnrType'] = X['MasVnrType'].fillna(self.most_frequent_)
        return X


class FillMasVnrArea(BaseEstimator, TransformerMixin):
    """Fill MasVnrArea with 0."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'MasVnrArea' in X.columns:
            X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        return X


class FillElectrical(BaseEstimator, TransformerMixin):
    """Fill Electrical with most frequent value."""
    
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if 'Electrical' in X.columns:
            mode_result = X['Electrical'].mode()
            self.most_frequent_ = mode_result[0] if len(mode_result) > 0 else 'SBrkr'
        else:
            self.most_frequent_ = 'SBrkr'
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'Electrical' in X.columns:
            X['Electrical'] = X['Electrical'].fillna(self.most_frequent_)
        return X


class QualMappingTransformer(BaseEstimator, TransformerMixin):
    """Convert categorical features to numerical using qualitative mappings."""
    
    def __init__(self):
        # Quality mappings (None, Po, Fa, TA, Gd, Ex)
        self.qual_mapping = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        
        # Specific feature mappings
        self.bsmt_exposure_mapping = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
        self.bsmt_fin_type_mapping = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
        self.functional_mapping = {'Maj2': 0, 'Maj1': 1, 'Mod': 2, 'Min2': 3, 'Min1': 4, 'Typ': 5}
        self.paved_drive_mapping = {'N': 0, 'P': 1, 'Y': 2}
        self.saletype_mapping = {'CWD': 0, 'ConLI': 1, 'ConLD': 2, 'COD': 3, 'New': 4, 'WD': 5}
        self.salecondition_mapping = {'Partial': 0, 'Family': 1, 'Alloca': 2, 'AdjLand': 3, 'Abnorml': 4, 'Normal': 5}
        self.heating_mapping = {'Floor': 1, 'OthW': 2, 'Wall': 3, 'Grav': 4, 'GasW': 5, 'GasA': 6}
        self.misc_feature_mapping = {'None': 0, 'Shed': 1, 'Othr': 2, 'Gar2': 5, 'TenC': 10}
        self.garage_type_mapping = {'Basment': 0, 'CarPort': 1, '2Types': 2, 'BuiltIn': 3, 'Detchd': 4, 'Attchd': 5}
        self.garage_finish_mapping = {'Unf': 1, 'RFn': 2, 'Fin': 3}
        self.electrical_mapping = {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}
        self.ms_zoning_mapping = {'C (all)': 0, 'RH': 1, 'RM': 2, 'RL': 3, 'FV': 4}
        self.street_mapping = {'Grvl': 0, 'Pave': 1}
        self.alley_mapping = {'None': 0, 'Grvl': 1, 'Pave': 2}
        self.lot_shape_mapping = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
        self.land_contour_mapping = {'Low': 0, 'Bnk': 1, 'HLS': 2, 'Lvl': 3}
        self.utilities_mapping = {'NoSeWa': 0, 'AllPub': 1}
        self.lot_config_mapping = {'FR3': 0, 'FR2': 1, 'Inside': 2, 'Corner': 3, 'CulDSac': 4}
        self.land_slope_mapping = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
        self.bsmt_fin_type2_mapping = {'Unf': 0, 'Rec': 1, 'LwQ': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5, 'None': 0}
        self.central_air_mapping = {'N': 0, 'Y': 1}
        self.fence_mapping = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        self.mas_vnr_type_mapping = {'None': 0, 'BrkCmn': 1, 'BrkFace': 2, 'Stone': 3}
    
    @staticmethod
    def misc_val_mapping(val):
        """Map MiscVal to categorical bins."""
        if pd.isna(val) or val == 0:
            return 0
        elif val < 1000:
            return 1
        elif val < 5000:
            return 2
        else:
            return 3
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply all categorical mappings."""
        df = X.copy()
        
        # Quality columns
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                        'PoolQC', 'FireplaceQu', 'GarageQual', 'GarageCond', 'KitchenQual']
        for col in quality_cols:
            if col in df.columns:
                df[col] = df[col].map(self.qual_mapping).fillna(0)
        
        # Specific feature mappings
        mappings = {
            'MSZoning': self.ms_zoning_mapping,
            'Street': self.street_mapping,
            'Alley': self.alley_mapping,
            'LotShape': self.lot_shape_mapping,
            'LandContour': self.land_contour_mapping,
            'Utilities': self.utilities_mapping,
            'LotConfig': self.lot_config_mapping,
            'LandSlope': self.land_slope_mapping,
            'BsmtExposure': self.bsmt_exposure_mapping,
            'BsmtFinType1': self.bsmt_fin_type_mapping,
            'BsmtFinType2': self.bsmt_fin_type2_mapping,
            'CentralAir': self.central_air_mapping,
            'Fence': self.fence_mapping,
            'MasVnrType': self.mas_vnr_type_mapping,
            'Functional': self.functional_mapping,
            'PavedDrive': self.paved_drive_mapping,
            'SaleType': self.saletype_mapping,
            'SaleCondition': self.salecondition_mapping,
            'Heating': self.heating_mapping,
            'MiscFeature': self.misc_feature_mapping,
            'GarageFinish': self.garage_finish_mapping,
            'Electrical': self.electrical_mapping
        }
        
        for col, mapping in mappings.items():
            if col in df.columns:
                if col == 'GarageType':
                    df[col] = df[col].fillna('None').map(self.garage_type_mapping).fillna(0)
                else:
                    df[col] = df[col].map(mapping).fillna(0)
        
        # MiscVal binning
        if 'MiscVal' in df.columns:
            df['MiscVal'] = df['MiscVal'].apply(self.misc_val_mapping)
        
        return df


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop specified features from dataset."""
    
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        cols_to_drop = [col for col in self.features_to_drop if col in df.columns]
        return df.drop(columns=cols_to_drop, errors='ignore')

