# House Prices - Advanced Regression Techniques

A professional machine learning project for predicting house prices using advanced regression techniques, feature engineering, and automated model optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a comprehensive machine learning pipeline for predicting house sale prices based on various property features. The solution leverages:

- **Advanced Feature Engineering**: Custom transformers and feature creation
- **Automated Model Selection**: TPOT for automated pipeline optimization
- **Ensemble Methods**: XGBoost with hyperparameter tuning
- **Robust Preprocessing**: Strategic missing value imputation and categorical encoding
- **Cross-Validation**: 10-fold CV for reliable performance estimation

## ✨ Features

- **Modular Architecture**: Clean, maintainable code structure
- **Pipeline-Based Preprocessing**: Reusable transformers and pipelines
- **Multiple Model Support**: XGBoost and TPOT models
- **Configuration Management**: YAML-based configuration
- **Comprehensive Logging**: Detailed training and prediction logs
- **Data Leakage Prevention**: Proper target encoding with smoothing

## 📁 Project Structure

```
House Prices/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── Raw/                      # Raw data files
│   └── Processed/                # Processed data files
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks (legacy)
├── reports/                      # Analysis reports
├── scripts/
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction script
├── src/
│   ├── preprocessing/
│   │   ├── transformers.py        # Custom transformers
│   │   └── pipelines.py           # Preprocessing pipelines
│   ├── features/
│   │   └── engineering.py          # Feature engineering
│   ├── models/
│   │   └── training.py            # Model training functions
│   └── utils/
│       └── helpers.py             # Utility functions
├── submission/                    # Submission files
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd "House Prices"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training a Model

Train an XGBoost model:
```bash
python scripts/train.py --model xgb
```

Train a TPOT model:
```bash
python scripts/train.py --model tpot
```

Train both models:
```bash
python scripts/train.py --model both
```

Custom configuration:
```bash
python scripts/train.py --config config/config.yaml --model xgb
```

### Making Predictions

Generate predictions using XGBoost:
```bash
python scripts/predict.py --model xgb
```

Generate predictions using TPOT:
```bash
python scripts/predict.py --model tpot
```

Custom paths:
```bash
python scripts/predict.py --model xgb --test-data data/Raw/test.csv --output submission/predictions.csv
```

## 🔬 Methodology

### 1. Data Preprocessing

#### Missing Value Imputation
- **LotFrontage**: Median imputation by neighborhood
- **Garage features**: Filled with 'None' (no garage)
- **Basement features**: Filled with 'None' (no basement)
- **MasVnrType/Electrical**: Most frequent value imputation
- **MasVnrArea**: Filled with 0

#### Categorical Encoding
- **Qualitative Mapping**: Ordinal encoding for quality-related features (None→0, Po→1, Fa→2, TA→3, Gd→4, Ex→5)
- **Custom Mappings**: Domain-specific mappings for all categorical features
- **No One-Hot Encoding**: All categoricals converted to numerical representations

### 2. Feature Engineering

#### New Features Created
- `TotalFinishedBsmtSF`: Sum of BsmtFinSF1 and BsmtFinSF2
- `TotalFinishedBsmtSF_BsmtQual_Interaction`: Interaction term
- `TotalFullBaths`: Sum of basement and above-ground full baths
- `TotalHalfBaths`: Sum of basement and above-ground half baths
- `HouseAge`: Years since construction (YrSold - YearBuilt)
- `YearsSinceRemodel`: Years since last remodel
- `TotalPorchArea`: Sum of all porch types

#### Feature Removal
- Original component features (redundant after aggregation)
- Weakly correlated features (Condition1, Condition2, BldgType, etc.)

### 3. Target Encoding

- **Neighborhood Encoding**: Smoothed target encoding to prevent data leakage
- **Smoothing Parameter**: α=10 to balance between neighborhood mean and global mean
- **Cross-Validation Safe**: Encoding calculated separately for each CV fold

### 4. Model Training

#### XGBoost
- Hyperparameter tuning with Optuna
- 10-fold cross-validation
- Outlier removal using Z-score (threshold=3)
- Log-transformed target (log1p)

#### TPOT
- Automated pipeline optimization
- Genetic programming for feature selection and model selection
- Stacking and ensemble methods

### 5. Evaluation

- **Metrics**: MSE, RMSE, MAE, R²
- **Cross-Validation**: 10-fold CV for robust performance estimation
- **Outlier Handling**: Z-score based filtering during training

## 📊 Results

Model performance is evaluated using 10-fold cross-validation:

### XGBoost Model
- Average RMSE: [To be filled after training]
- Average MAE: [To be filled after training]
- Average R²: [To be filled after training]

### TPOT Model
- Average RMSE: [To be filled after training]
- Average MAE: [To be filled after training]
- Average R²: [To be filled after training]

## ⚙️ Configuration

All parameters can be configured in `config/config.yaml`:

- Data paths
- Model hyperparameters
- Cross-validation settings
- Feature engineering options
- Logging configuration

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running scripts from the project root directory
2. **Missing Data**: Ensure data files are in the correct paths specified in config.yaml
3. **Model Not Found**: Train the model first using `scripts/train.py`

## 📝 Notes

- The project uses log-transformed targets (log1p) for better model performance
- All categorical features are converted to numerical using qualitative mappings
- Neighborhood encoding is calculated separately for each CV fold to prevent data leakage
- Outliers are removed during training but predictions are made on all test data

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Kaggle for providing the dataset and competition platform
- TPOT developers for the automated ML library
- XGBoost team for the powerful gradient boosting framework

---

**Note**: This project is based on the Kaggle competition "House Prices - Advanced Regression Techniques".
