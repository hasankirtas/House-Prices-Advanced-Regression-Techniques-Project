"""
Prediction script for House Prices model.
Generates predictions on test data and creates submission file.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipelines import create_full_preprocessing_pipeline
from src.utils.helpers import apply_neighborhood_encoding
from sklearn.impute import SimpleImputer

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


def preprocess_test_data(config, test_path):
    """
    Preprocess test data using the same pipeline as training.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    test_path : str
        Path to test data CSV
    
    Returns
    -------
    pd.DataFrame
        Preprocessed test data
    """
    logger.info(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path)
    
    # Save Id column for submission
    if 'Id' in test_data.columns:
        test_ids = test_data['Id'].copy()
        test_data.drop(columns=['Id'], inplace=True)
    else:
        test_ids = None
    
    logger.info(f"Test data shape: {test_data.shape}")
    
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
    
    # Apply preprocessing (fit_transform to learn from test data)
    logger.info("Applying preprocessing...")
    test_data_processed = pipeline.fit_transform(test_data)
    
    # Restore Neighborhood if needed for encoding
    if 'Neighborhood' not in test_data_processed.columns:
        # Try to get it from original test data
        if 'Neighborhood' in test_data.columns:
            test_data_processed['Neighborhood'] = test_data['Neighborhood'].values
    
    # Save processed data
    processed_path = config['data']['processed']['test']
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    test_data_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed test data saved to {processed_path}")
    
    return test_data_processed, test_ids


def load_neighborhood_encoding(encoding_path):
    """
    Load neighborhood encoding from CSV file.
    
    Parameters
    ----------
    encoding_path : str
        Path to encoding CSV file
    
    Returns
    -------
    dict
        Neighborhood to smoothed price mapping
    float
        Global mean (estimated from encoding)
    """
    encoding_df = pd.read_csv(encoding_path)
    encoding_dict = dict(zip(encoding_df['Neighborhood'], encoding_df['smoothed']))
    global_mean = encoding_df['smoothed'].mean()  # Approximate global mean
    return encoding_dict, global_mean


def make_predictions(
    test_data,
    model_path,
    encoding_path,
    config
):
    """
    Make predictions on test data.
    
    Parameters
    ----------
    test_data : pd.DataFrame
        Preprocessed test data
    model_path : str
        Path to trained model
    encoding_path : str
        Path to neighborhood encoding
    config : dict
        Configuration dictionary
    
    Returns
    -------
    np.ndarray
        Predictions (in original scale, not log)
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading neighborhood encoding from {encoding_path}")
    neighborhood_encoding, global_mean = load_neighborhood_encoding(encoding_path)
    
    # Apply neighborhood encoding
    if 'Neighborhood' in test_data.columns:
        test_data_encoded = apply_neighborhood_encoding(
            test_data, neighborhood_encoding, global_mean
        )
    else:
        test_data_encoded = test_data.copy()
        logger.warning("Neighborhood column not found, skipping encoding")
    
    # Handle any remaining missing values
    # Use most_frequent strategy (not 'mod' which is invalid)
    logger.info("Handling remaining missing values...")
    imputer = SimpleImputer(strategy='most_frequent')
    
    # Fit on test data and transform
    test_data_imputed = pd.DataFrame(
        imputer.fit_transform(test_data_encoded),
        columns=test_data_encoded.columns,
        index=test_data_encoded.index
    )
    
    # Make predictions (model expects log-transformed target)
    logger.info("Making predictions...")
    predictions_log = model.predict(test_data_imputed)
    
    # Convert back to original scale
    predictions = np.expm1(predictions_log)
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions


def create_submission(predictions, test_ids, output_path):
    """
    Create submission file.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predictions array
    test_ids : pd.Series or None
        Test IDs
    output_path : str
        Path to save submission file
    """
    logger.info("Creating submission file...")
    
    if test_ids is None:
        # Create sequential IDs if not provided
        test_ids = pd.Series(range(1, len(predictions) + 1))
        logger.warning("Test IDs not found, using sequential IDs")
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Generate predictions on test data')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgb', 'tpot'],
        default='xgb',
        help='Model to use for predictions (xgb or tpot)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test data CSV (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output submission file (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths
    test_path = args.test_data if args.test_data else config['data']['raw']['test']
    model_path = config['models'][args.model]
    encoding_path = config['models']['neighborhood_encoding']
    output_path = args.output if args.output else config['submission']['output']
    
    # Preprocess test data
    test_data, test_ids = preprocess_test_data(config, test_path)
    
    # Make predictions
    predictions = make_predictions(
        test_data,
        model_path,
        encoding_path,
        config
    )
    
    # Create submission file
    create_submission(predictions, test_ids, output_path)
    
    logger.info("\nPrediction completed successfully!")


if __name__ == '__main__':
    main()

