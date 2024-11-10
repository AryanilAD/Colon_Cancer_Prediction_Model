# Colon Cancer Prediction Model

This project aims to build a machine learning model for predicting types of colon cancer using two datasets merged on a common identifier. Each step of the process is explained in detail, from data loading and preprocessing to model training, evaluation, and testing on unseen data.

## Table of Contents
- [Project Overview](#project-overview)
- [Workflow](#workflow)
  - [1. Data Loading](#1-data-loading)
  - [2. Data Merging](#2-data-merging)
  - [3. Data Inspection](#3-data-inspection)
  - [4. Data Preparation and Transformation](#4-data-preparation-and-transformation)
  - [5. Feature Selection and Feature Importance](#5-feature-selection-and-feature-importance)
  - [6. Model Training and Evaluation](#6-model-training-and-evaluation)
  - [7. Prediction and Testing on Unseen Data](#7-prediction-and-testing-on-unseen-data)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Project Overview

This project combines two colon cancer datasets for predicting cancer types. By merging and transforming the datasets, applying feature engineering, and training various machine learning models, this project aims to create a predictive model for colon cancer.

## Workflow

### 1. Data Loading

In this step, we load the two datasets into pandas DataFrames and conduct a preliminary inspection to understand the data structure.

```python
import pandas as pd
df1 = pd.read_csv("dataset_1_colon_cancer.csv")
df2 = pd.read_csv("dataset_2_colon_cancer.csv")
```

**Explanation**: Initial inspection of the datasets helps to identify the key columns and understand which columns may contain overlapping or complementary information.

*Screenshot: Displaying initial data preview and column names.*

### 2. Data Merging

The datasets are merged using an outer join on the common column, `Type of Colon Cancer`. This ensures that no data is lost from either dataset, resulting in a complete merged dataset.

```python
merged_df = pd.merge(df1, df2, on='Type of Colon Cancer', how='outer')
```

**Explanation**: Merging is crucial because it aligns the data from both datasets on a single axis, enhancing our model by pooling all relevant information into one dataset. An outer join is selected to retain all data points, even if some are missing in one dataset, ensuring maximum data availability.

*Screenshot: Overview of the merged dataset structure and summary statistics.*

### 3. Data Inspection

After merging, we inspect the data types, null values, and general structure of the dataset.

```python
print(merged_df.info())
```

**Explanation**: Inspecting the data allows us to identify columns with null values or unexpected data types. This step informs later data preprocessing choices, such as imputation or type conversion.

*Screenshot: Data types and null value report.*

### 4. Data Preparation and Transformation

This step includes handling missing values, encoding categorical features, and scaling numerical features for better model performance.

#### Handling Missing Values
Missing values are addressed by either imputing or removing them, depending on their impact on the dataset.

```python
merged_df.fillna(merged_df.mean(), inplace=True)  # Example of mean imputation for numerical data
```

#### Encoding Categorical Variables
Categorical features are converted to numerical format using one-hot encoding.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded_categorical = encoder.fit_transform(merged_df[categorical_columns])
```

#### Scaling Numerical Features
Numerical features are scaled for consistent data ranges, benefiting model training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_df[numerical_columns])
```

**Explanation**: Preprocessing ensures the data is clean and suitable for model training. Encoding and scaling create uniform feature representations, which aids the model in learning patterns more effectively.

*Screenshot: Example of encoded and scaled data.*

### 5. Feature Selection and Feature Importance

Feature importance is determined to identify which attributes have the most predictive power. This can be done using feature importance scores from models like Random Forest or by correlation analysis.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
importance = model.feature_importances_
```

**Explanation**: Feature importance helps to reduce dimensionality by removing less relevant features, improving model efficiency and interpretability. In this project, important features include `AGE`, `CEA Level`, and `Tumor Grade`.

*Screenshot: Visualization of feature importance scores.*

### 6. Model Training and Evaluation

The data is split into training and testing sets, and multiple models are trained, including Logistic Regression, Random Forest, and Support Vector Machine (SVM). The best model is selected based on performance metrics such as accuracy, precision, recall, and F1-score.

#### Splitting Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Model Training and Evaluation
```python
from sklearn.metrics import classification_report

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

**Explanation**: By training multiple models, we can compare their performance and choose the most effective one. Evaluating on the test set provides insight into the model's ability to generalize to unseen data.

*Screenshot: Classification report and performance metrics.*

### 7. Prediction and Testing on Unseen Data

After model evaluation, we test the final model on an unseen dataset to confirm its predictive capabilities. This simulates real-world performance, ensuring the model is robust beyond the training data.

```python
# Load unseen test data
unseen_df = pd.read_csv("unseen_data.csv")

# Preprocess and predict
unseen_processed = scaler.transform(unseen_df[numerical_columns])  # Example transformation
unseen_predictions = model.predict(unseen_processed)
```

**Explanation**: Testing on unseen data is crucial to assess the model's performance in real-world scenarios. This final step confirms the model's predictive strength when faced with data it has not encountered before.

*Screenshot: Prediction results on unseen data.*

---

## Results

The model’s final performance metrics are summarized here. Testing on unseen data validates the model's reliability and accuracy in predicting colon cancer types.

## Conclusion

This project demonstrates a thorough machine learning workflow, from data merging and preprocessing to feature engineering and model testing. The final model holds promise for aiding in early colon cancer prediction, leveraging combined datasets for comprehensive insights.
