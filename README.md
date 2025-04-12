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

#### **2. Feature Engineering**

Various feature engineering techniques were applied to enhance the model's predictive power:

1. **Average Price Calculation**
   - `Neighborhood_avg_price`: The average house sale price for each neighborhood was calculated and added to the test data to capture neighborhood-specific pricing trends.

2. **Categorical Variable Conversion to Numeric**
   - `GarageType_num`: The categorical variable `GarageType` was transformed into numeric values.
   - `GarageFinish_num`: The categorical variable `GarageFinish` was transformed into numeric values.
   - `BsmtExposure_Score`: The categorical variable `BsmtExposure` was transformed into numeric values.

3. **Missing Value Imputation**
   - `GarageYrBlt`: Missing values were imputed with 0 to represent houses with no garage year information.
   - `GarageType_num`, `GarageFinish_num`, `Garage_Feature`, `Living_Area_per_Room`, `Garage_Capacity_per_Square_Meter`: Missing values in these columns were imputed with the respective column's mean value.

4. **New Feature Creation**
   - `Garage_Feature`: A new feature was created by weighting different garage-related characteristics (such as vehicle capacity, area, finish status, and year built).
   - `Living_Area_per_Room`: The total living area was divided by the total number of rooms to create a new feature representing the average living area per room.
   - `Garage_Capacity_per_Square_Meter`: The garage capacity was divided by the garage area to create a new feature representing garage capacity per square meter.
   - `GrLivArea_per_Room`: The total living area was divided by the total number of rooms to create a feature representing the average living area per room.
   - `MasVnr_Area_to_TotalArea`: The ratio of masonry veneer area to total area was calculated to understand the proportion of masonry in the house.
   - `Age_at_Remodel`: The age of the house at the time of remodeling was calculated to capture how old the house was when it was last renovated.
   - `BsmtFinSF_to_TotalArea`: The ratio of finished basement area to total area was calculated to determine the impact of basement space on overall area.
   - `Lot_Frontage_to_Area`: The ratio of lot frontage to total area was calculated to understand the proportion of the lot's frontage in relation to its total area.
   - `TotalOutdoorArea`: A new feature was created by summing all outdoor areas (such as terrace, second floor, porch) to capture the total outdoor space available.
   - `Overall_Quality`: The sum of various quality attributes of the house was used to create a general quality score for each property.
   - `FullBath_to_Bedrooms`: The ratio of full bathrooms to bedrooms was calculated to measure the balance between bathrooms and bedrooms.
   - `Fireplace_Impact`: The number of fireplaces and the quality score of each fireplace were multiplied to calculate the overall impact of fireplaces on the property.
   - `BsmtQual_to_BsmtFinSF`: The basement quality score was multiplied by the finished basement area to create a feature representing the overall basement quality and finish.
   - `Overall_Quality_Impact`: The general quality score of the house was multiplied by the living area to create a new feature representing the combined impact of the house's quality and size.

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

### File Structure
```plaintext
Project/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── test_data_processing.ipynb
│   ├── Final_Test_Predictions.ipynb
├── src/
│   ├── data_preprocessing.py
├── reports/
├── README.md
├── environment.yml
├── requirements.txt
```

---

### Installation
To set up the project environment:

Using Conda:
```bash
conda env create -f environment.yml
conda activate House-Prices-Advanced-Regression-Techniques-Project
```

Using Pip:
```bash
pip install -r requirements.txt
```

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
