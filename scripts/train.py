"""
Training script for House Prices prediction model.
Supports both XGBoost and TPOT models.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipelines import create_full_preprocessing_pipeline
from src.models.training import train_xgboost_with_cv, train_tpot_with_cv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def preprocess_data(config, train_path):
    """
    Preprocess training data.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    train_path : str
        Path to training data CSV
    
    Returns
    -------
    pd.DataFrame
        Preprocessed training data
    """
    logger.info(f"Loading training data from {train_path}")
    train_data = pd.read_csv(train_path)
    
    # Drop Id column if present
    if 'Id' in train_data.columns:
        train_data.drop(columns=['Id'], inplace=True)
    
    logger.info(f"Training data shape: {train_data.shape}")
    
    # Get features to drop from config
    features_to_drop = (
        config['features_to_drop']['original_components'] +
        config['features_to_drop']['weakly_related']
    )
    
    # Create preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")
    pipeline = create_full_preprocessing_pipeline(
        features_to_drop=features_to_drop,
        include_bsmt_interaction=config['feature_engineering']['include_bsmt_quality_interaction']
    )
    
    # Apply preprocessing
    logger.info("Applying preprocessing...")
    train_data_processed = pipeline.fit_transform(train_data)
    
    # Save processed data
    processed_path = config['data']['processed']['train']
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    train_data_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    return train_data_processed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train House Prices prediction model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgb', 'tpot', 'both'],
        default='xgb',
        help='Model to train (xgb, tpot, or both)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data CSV (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get data path
    train_path = args.data if args.data else config['data']['raw']['train']
    
    # Preprocess data
    train_data = preprocess_data(config, train_path)
    
    # Prepare features and target
    X = train_data.drop(columns=['SalePrice'])
    y = np.log1p(train_data['SalePrice'])  # Log transform target
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Train models
    if args.model in ['xgb', 'both']:
        logger.info("\n" + "="*50)
        logger.info("Training XGBoost Model")
        logger.info("="*50)
        
        train_xgboost_with_cv(
            X, y, config,
            save_model_path=config['models']['xgb'],
            save_encoding_path=config['models']['neighborhood_encoding']
        )
    
    if args.model in ['tpot', 'both']:
        logger.info("\n" + "="*50)
        logger.info("Training TPOT Model")
        logger.info("="*50)
        
        train_tpot_with_cv(
            X, y, config,
            save_model_path=config['models']['tpot'],
            save_encoding_path=config['models']['neighborhood_encoding']
        )
    
    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

