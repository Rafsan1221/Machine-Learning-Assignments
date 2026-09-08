# Problem Set 02: Bank Term Deposit Subscription Prediction

## 1. Approach

This project builds a Logistic Regression model to predict whether a bank customer will subscribe to a term deposit (`yes`/`no`), using demographic and behavioral features from the Bank Marketing dataset.

## 2. Methodology

### Data Loading & Preprocessing
- Extracted data from `bank-data.zip` (`bank-full.csv`).
- Converted the target variable `y` into binary form — `1` for `yes`, `0` for `no`.
- Applied **One-Hot Encoding** (`OneHotEncoder(drop='first')`) to categorical features.
- Standardized numerical features using `StandardScaler`.
- Combined both steps inside a `ColumnTransformer`, wrapped in a `Pipeline`, to prevent data leakage between training and testing sets.

### Model Selection & Handling Class Imbalance
- Used `LogisticRegression` with `class_weight='balanced'` to address the dataset's skew toward non-subscribers.
- Split the data 80% training / 20% testing, using stratified sampling (`random_state=42`) to preserve class distribution.

## 3. Findings & Performance Metrics

| Metric | Score |
|---|---|
| Accuracy | ~84.6% |
| ROC-AUC | ~0.908 |

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (Non-subscribers) | 0.97 | 0.85 | 0.91 |
| 1 (Subscribers) | 0.42 | 0.81 | 0.55 |

### Insights
- Balancing class weights allowed the model to correctly identify **81% of actual subscribers** (high recall for Class 1) — critical for targeted marketing campaigns.
- Despite lower precision on Class 1 due to class imbalance, the model achieves strong overall discriminatory ability, reflected in a **ROC-AUC above 0.90**.
