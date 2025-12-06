"""
Data preprocessing pipelines.
Combines all transformers into reusable pipelines.
"""

from sklearn.pipeline import Pipeline
from src.preprocessing.transformers import (
    FillNoneCols,
    FillLotFrontageByNeighborhood,
    FillGarageCols,
    FillBsmtCols,
    FillMasVnrType,
    FillMasVnrArea,
    FillElectrical,
    QualMappingTransformer,
    FeatureDropper
)
from src.features.engineering import FeatureEngineeringTransformer


def create_imputation_pipeline():
    """Create pipeline for missing value imputation."""
    none_fill_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    
    pipeline = Pipeline([
        ('fill_none_cols', FillNoneCols(none_fill_cols)),
        ('fill_lot_frontage', FillLotFrontageByNeighborhood()),
        ('fill_garage_cols', FillGarageCols()),
        ('fill_bsmt_cols', FillBsmtCols()),
        ('fill_masvnr_type', FillMasVnrType()),
        ('fill_masvnr_area', FillMasVnrArea()),
        ('fill_electrical', FillElectrical())
    ])
    
    return pipeline


def create_full_preprocessing_pipeline(features_to_drop, include_bsmt_interaction=True):
    """
    Create complete preprocessing pipeline.
    
    Parameters
    ----------
    features_to_drop : list
        List of feature names to drop after feature engineering
    include_bsmt_interaction : bool, default=True
        Whether to include basement quality interaction term
    
    Returns
    -------
    Pipeline
        Complete preprocessing pipeline
    """
    pipeline = Pipeline([
        ('feature_imputation', create_imputation_pipeline()),
        ('qual_mapping', QualMappingTransformer()),
        ('engineer_features', FeatureEngineeringTransformer(
            include_bsmt_quality_interaction=include_bsmt_interaction
        )),
        ('drop_features', FeatureDropper(features_to_drop=features_to_drop))
    ])
    
    return pipeline

