
# Colon Cancer Prediction Model

This project involves developing a machine learning model to predict a continuous target variable related to colon cancer metrics. It integrates data from two separate datasets, performs extensive data preprocessing, and applies predictive modeling to evaluate the effectiveness of various machine learning algorithms. The project follows a structured workflow that includes data loading, merging, transformation, feature selection, model training, evaluation, and testing on unseen data. Key steps are illustrated with screenshots to enhance clarity.

## Table of Contents
- [Project Overview](#project-overview)
- [Workflow](#workflow)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Merging](#2-data-merging)
  - [3. Data Inspection](#3-data-inspection)
  - [4. Data Preparation and Transformation](#4-data-preparation-and-transformation)
  - [5. Feature Engineering and Feature Selection](#5-feature-engineering-and-feature-selection)
  - [6. Model Training and Evaluation](#6-model-training-and-evaluation)
  - [7. Prediction and Testing on Unseen Data](#7-prediction-and-testing-on-unseen-data)
- [Results](#results)
- [Conclusion](#conclusion)
- [Adding Screenshots to README](#adding-screenshots-to-readme)

---

## Project Overview

The objective of this project is to create a predictive model for colon cancer by utilizing two datasets. After merging and transforming the datasets, the project applies feature engineering techniques and trains multiple regression models to predict a target value indicative of colon cancer progression. The model is then evaluated using prediction-specific metrics, and its performance is validated on unseen data to ensure robustness.

## Workflow

### 1. Data Loading

The project begins by loading the two datasets. Each dataset is loaded into a pandas DataFrame and inspected to understand its structure, identify key columns, and determine the initial data quality. During this step, we review column names, check data types, and inspect the first few rows of each dataset to establish the data’s general composition.

### 2. Data Merging

The datasets are merged on a common column, `Type of Colon Cancer`, using an outer join. This method is chosen to ensure that all unique rows from both datasets are retained, regardless of whether they contain matching values in the common column. This outer join approach helps maximize the available data, which is especially important for a predictive task where each additional data point can contribute to model accuracy. After merging, the resulting dataset is inspected to confirm that the merge was successful and to evaluate the structure and size of the combined data.

![Screenshot 2024-11-10 151335](https://github.com/user-attachments/assets/9f3fb390-9ab2-49a9-8a4b-b83eb57668f4)

### 3. Data Inspection

With the merged dataset, an in-depth data inspection is conducted. This includes examining data types, checking for null values, and performing statistical analysis on numerical columns to understand their distributions. Special attention is paid to any null values, as handling missing data will be crucial for accurate model training. This step provides insights into any necessary data cleaning or transformation that will be addressed in subsequent steps.

![Screenshot 2024-11-10 151411](https://github.com/user-attachments/assets/38b57be8-7791-42cd-be06-1d1d0f7ad731)


### 4. Data Preparation and Transformation

This critical step involves preparing and transforming the dataset to ensure that it is suitable for model training. Key tasks include:

- **Handling Missing Values**: Missing values in numerical columns are imputed using statistical methods such as mean or median imputation. For categorical columns, missing values are addressed through imputation techniques suited to the column’s characteristics or by dropping columns if they contain excessive missing values.
  
- **Encoding Categorical Variables**: Categorical variables are converted into a numerical format. For instance, one-hot encoding is applied to columns with multiple categories, resulting in binary columns for each category. This step is essential for regression models that cannot interpret categorical data directly.
  
- **Scaling Numerical Features**: Numerical features are scaled using standardization or normalization to ensure uniform data ranges. This is particularly important in predictive modeling, as it prevents features with larger ranges from disproportionately influencing the model’s predictions.

  ![Screenshot 2024-11-10 151355](https://github.com/user-attachments/assets/22ad3b6a-4501-4b32-9326-f33d53c8683e)


After data preparation, the dataset is re-inspected to confirm that all transformations have been applied correctly. At this point, the data is clean, with categorical features encoded and numerical features scaled, making it ready for feature engineering.

### 5. Feature Importance Analysis

**Feature Selection**: After generating potential new features, we evaluate each feature’s importance to identify those with the highest predictive value. Techniques such as correlation analysis and feature importance from models like Random Forest are used to rank features. Based on this analysis, non-informative features are dropped to reduce dimensionality and improve model efficiency.

![Screenshot 2024-11-10 151308](https://github.com/user-attachments/assets/18e94c7b-ff67-49a6-83a3-293eb07ab254) 

We can see from the above image variables like CEA Level, Polyp Size (mm) and AGE are showing a significant importance factor in predicting the cancer type.

The resulting set of features balances predictive power with simplicity, aiming to maximize model performance without unnecessary complexity.

### 6. Model Training and Evaluation

With a refined dataset, we split the data into training and testing sets. This split allows for model evaluation on data it hasn’t seen during training, giving an unbiased estimate of model performance.

- **Model Selection**: Two regression models are tested, including Linear Regression, and Random Forest Regressor. These models are chosen for their different approaches to capturing relationships within the data.
  
- **Training**: Each model is trained on the training set using the selected features. Hyperparameter tuning is performed to optimize the model’s predictive accuracy.

<h3>Random Forest Regressor</h3>

![Screenshot 2024-11-11 105102](https://github.com/user-attachments/assets/6794d5cc-2f87-49c3-ab09-69d7b574276a)

<h3>Linear Regression</h3>

![Screenshot 2024-11-11 105042](https://github.com/user-attachments/assets/c3a042b4-c871-41dd-9cee-f9107646ca98)

- **Evaluation Metrics**: Since this is a prediction task, models are evaluated using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), which provide insight into the model’s accuracy in predicting continuous values. RMSE is particularly useful here as it penalizes larger errors, which is important in medical predictions where accuracy is critical.

Each model’s performance on the test set is compared to identify the best-performing model. RMSE and MAE scores are recorded for each model to aid in final selection.

### 7. Prediction and Testing on Unseen Data

The final, both the model is then tested on an unseen dataset. This step simulates real-world deployment and provides a realistic view of the model’s predictive capabilities when applied to new data. The unseen data undergoes the same preprocessing steps as the training data to ensure consistency. I have created a new unseen data which is located into the unseen csv file to check the prediction of the model. in my analysis Random Forrest Regressor is working better on unseen data rather than linear Regression.

## Results

After completing model training and evaluation, the following metrics were recorded for the final model:

![image](https://github.com/user-attachments/assets/24758232-7b6a-4243-9d59-22bd0b893361)


These metrics demonstrate the model’s effectiveness in predicting the target variable accurately. The low RMSE and MAE values indicate that the model is well-calibrated, while a high R-squared suggests strong explanatory power.

## Conclusion

This project showcases a structured machine learning pipeline, from data merging and feature engineering to model training and validation. The final model holds promise for aiding in predictive analytics for colon cancer, leveraging the combined power of two datasets and multiple regression techniques.

