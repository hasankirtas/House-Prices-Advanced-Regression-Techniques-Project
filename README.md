# Project: House Prices - Advanced Regression Techniques

### Overview
This project aims to build a machine learning model that predicts house prices based on a variety of features. The dataset is based on the Kaggle competition "House Prices - Advanced Regression Techniques." The goal is to predict the final sale price of houses using data preprocessing, feature engineering, and model optimization techniques. The project also focuses on exploring various machine learning methods and improving the model's performance through stacking and hyperparameter tuning.

---

### Objectives
1. Develop a robust machine learning pipeline for regression tasks.
2. Implement effective feature engineering strategies.
3. Address missing values and outliers in the data.
4. Optimize model performance using advanced techniques like stacking and hyperparameter tuning.
5. Ensure model generalizability and prevent overfitting.

---

### Steps Followed

#### **1. Data Loading and Preprocessing**
- **Dataset**: The raw data was loaded from `../data/raw/train.csv`.
- **Target Variable**: The target variable `SalePrice` was extracted as the dependent variable for regression.
- **Preprocessing**: Missing values and outliers were handled to ensure clean data.
- **Feature Encoding**: Instead of using one-hot or label encoding, a qualitative mapping strategy (`qual-mapping`) was used to convert categorical variables into numerical values.
- **Train-Test Split**: Data was split into 80% training and 20% testing using a random state for reproducibility.

---

#### **Sweetviz for Data Analysis and Visualization**
- **Sweetviz** was used to perform an in-depth analysis of the dataset, providing interactive visualizations and summary reports for better understanding of the features and their relationships. It helped uncover important trends and patterns, which guided feature engineering and preprocessing decisions.
- Sweetviz generated clear and comprehensive reports, enabling quick identification of correlations, distributions, and missing values across different variables. These insights helped ensure that the data preparation was aligned with the overall goal of building an effective predictive model.

---

### **2. Feature Engineering**
---
Various feature engineering techniques were applied to enhance the model's predictive power, focusing on imputation, categorical to numerical conversions, new feature creation, and removal of redundant or less impactful features.

#### **Missing Value Imputation and Categorical to Numeric Conversion**

* **Strategic Imputation:** Missing values in various columns were handled using custom transformers:
    * `LotFrontage` was imputed with the **median `LotFrontage` for each `Neighborhood`**.
    * Garage-related columns (`GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`) had their missing values filled with **'None'**, and `GarageYrBlt` was filled with **0**.
    * Basement-related columns (`BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`) were imputed with **'None'**.
    * `PoolQC`, `MiscFeature`, `Alley`, `Fence`, and `FireplaceQu` were also filled with **'None'**.
    * `MasVnrType` and `Electrical` were imputed with their **most frequent values**.
    * `MasVnrArea` missing values were filled with **0**.
* **Extensive Categorical Mapping:** A `QualMappingTransformer` was implemented to convert a wide range of categorical features into numerical representations based on predefined mappings, reflecting ordinal relationships where applicable. This included:
    * Quality-related columns (e.g., `ExterQual`, `BsmtQual`, `FireplaceQu`, `KitchenQual`) mapped to numerical scores (e.g., 'None': 0, 'Po': 1, 'Ex': 5).
    * Specific categorical features like `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `CentralAir`, `Fence`, `MasVnrType`, `Functional`, `PavedDrive`, `GarageType`, `SaleType`, `SaleCondition`, `Heating`, `MiscFeature`, `GarageFinish`, and `Electrical` were also mapped to numerical values.
    * `MiscVal` was binned into numerical categories (0, 1, 2, 3) based on its value range.

#### **New Feature Creation**

* `TotalFinishedBsmtSF`: Calculated as the sum of `BsmtFinSF1` and `BsmtFinSF2`.
* `TotalFinishedBsmtSF_BsmtQual_Interaction`: An interaction term created by multiplying `TotalFinishedBsmtSF` with the **numerical `BsmtQual` score**.
* `TotalFullBaths`: Sum of `BsmtFullBath` and `FullBath`.
* `TotalHalfBaths`: Sum of `BsmtHalfBath` and `HalfBath`.
* `HouseAge`: Calculated as `YrSold` - `YearBuilt`.
* `YearsSinceRemodel`: Calculated as `YrSold` - `YearRemodAdd`, ensuring **non-negative values**.
* `TotalPorchArea`: Sum of `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, and `ScreenPorch`.

#### **Feature Dropping**

To reduce redundancy and simplify the model, several original features were dropped after their information was incorporated into new, engineered features. These include:

* **Original component features:** `BsmtFinSF1`, `BsmtFinSF2`, `BsmtFullBath`, `FullBath`, `BsmtHalfBath`, `HalfBath`, `YearBuilt`, `YearRemodAdd`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`.
* **Features with weak correlation** to the target variable: `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `Foundation`.

#### **3. Model Development**
- **TPOT**: The Stacked model was built using the TPOT technology, which automates the process of stacking and provides an optimized ensemble model.
- **Parameter Tuning**: TPOT also allowed for advanced hyperparameter optimization to improve the model's performance.
- **Final Model**: A highly optimized stacked model with multiple regression techniques was created to predict house prices.

#### **4. Model Evaluation**
- **Metrics**: The model was evaluated using common regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
- **Cross-validation**: 5-fold cross-validation was applied to assess the generalizability of the model and ensure stable performance across different subsets of the data.

#### **5. Overfitting**
- Overfitting was not observed in the model during the training and testing phases. Consequently, no additional steps were taken to address overfitting.

#### **6. Automation and Reporting**
- The model evaluation results were automatically saved in a report file for further analysis.
- **Report**: The report contains evaluation metrics and the model’s performance on various test subsets.

---

### Challenges and Solutions

1. **Handling Categorical Variables**:
   - Challenge: Converting categorical features into numeric values without one-hot or label encoding.
   - Solution: Applied a `qual-mapping` strategy to convert categorical features to numeric values based on their qualitative meaning.

2. **Feature Engineering**:
   - Challenge: The dataset contained a mix of features requiring transformations to enhance predictive power.
   - Solution: Multiple new features were created to capture important relationships within the data, such as the ratio of various areas and quality-related features.

3. **Model Optimization**:
   - Challenge: Finding an optimal regression model that performs well on the dataset.
   - Solution: Used TPOT for stacking models and hyperparameter tuning, resulting in an optimized ensemble model.

4. **Overfitting**:
   - Challenge: Overfitting was a potential concern during model training.
   - Solution: No overfitting was detected, and therefore, no specific steps were taken to address this issue.

---

### Tools and Libraries Used
- **Programming Language**: Python
- **Libraries**:
  - pandas, numpy (Data manipulation)
  - Sweetviz (Data analysis and visualization)
  - scikit-learn (Modeling and evaluation)
  - TPOT (Automated machine learning and stacking)
  - matplotlib, seaborn (Visualization)

---

### Conclusion
This project was a highly enjoyable experience, allowing for exploration of advanced regression techniques, feature engineering, and model optimization strategies. By leveraging TPOT for automated model stacking and hyperparameter tuning, a strong model for predicting house prices was developed. Future improvements could involve testing additional regression algorithms and fine-tuning the model further.

---

### Acknowledgements
Special thanks to Kaggle for providing the dataset and the inspiration for this challenge.

---

###  My Thoughts and Insights
This project significantly enhanced my skills in feature engineering and model optimization. Working with the TPOT library allowed me to explore automated machine learning techniques and improve the model with advanced stacking and parameter tuning. The process of creating new features and handling categorical variables with a qual-mapping strategy also gave me deeper insights into data transformation techniques.

---
