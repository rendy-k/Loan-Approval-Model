# Machine Leaning Models for Loan Approval

## 1. Exploratory Data Analysis (EDA)
### 1a. Data Duplicates

There are 58,645 samples in the "train" dataset.

There is no duplicated sample.

There is no missing value.

### 1b. Data Distribution
#### Numerical Features

The samples with loan_status = 1 contribute to 14.2% of the total samples. This is an imbalanced dataset.

The numerical features plot in the histograms show that all of the features are right-skewed. The 3 features having outliers are 'person_age', 'person_income', and 'person_emp_length'.

![alt text](images/numerical_features.png)

The samples with outliers are removed. 'Person_age' should not be more than 100 and 'person_emp_length' should not be more than 80. Meanwhile, outliers in 'person_income' still make sense.

![alt text](images/feature_boxplots.png)

#### Categorical Features

There are originally 4 categorical features. But, 'loan_grade' should not be included as a predictor feature, unless the grading is available before the applicant submits a loan.

![alt text](images/categorical_features.png)

### 1c. Train-test split

Twenty percent of the dataset is allocated for the validation set. The training and validation split is applied based on the label stratification. 

## 2. Feature Engineering

### 2a. Optimal Binning

Optimal binning is used to gain insights from the feature information values. The information value of each feature indicates how strongly an individual feature can predict the label. The table below shows how strong the predictive power of each feature.

Numerical Features:

| name     | n_bins    | iv | predictive power |
| -------- | -------- | ------- |  ------- |
| loan_percent_income  | 8  | 1.245736 | too good |
| loan_int_rate | 8 | 1.117075 | too good |
| person_income  | 8 | 0.608331 | too good |
| loan_amnt | 5  | 0.177468 | medium |
| person_emp_length | 7 | 0.128329 | medium |
| person_age | 3 | 0.005389 | useless |
| cb_person_cred_hist_length | 2 | 0.001252 | useless |

Categorical Features:

| name     | n_bins    | iv | predictive power |
| -------- | -------- | ------- |  ------- |
| loan_grade  | 4  | 1.238536 | too good |
| person_home_ownership | 3 | 0.601825 | too good |
| cb_person_default_on_file  | 2 | 0.225575 | medium |
| loan_intent | 6  | 0.095048 | weak |

### 2b. Feature Transformation

The optimal binning transformation is applied to loan_intent as it has the most number of unique values.

One-hot encoding is applied to 'person_home_ownership' and 'cb_person_default_on_file'.

### 2c. Predictive Power Score (PPS)

PPS aims to find which features have a strong relationship. Multiple features with strong relationships can be removed. PPS is the substitute for the commonly used correlation test. 

Running the PPS shows that there is no feature with a strong relationship with another feature. Feature selection is not needed.

## 3. Model Development and Evaluation

XGBoost algorithm is used to train the Machine Learning model. The hyperparameter-tuning is applied using Bayesian Optimization. A set of hyperparameters is gained by optimizing the ROC AUC.

The trained model is stable as indicated by the similarity between the training, testing, and validation ROC AUCs. But, the metrics of precision, f1-score, and accuracy of the training and testing sets are not close.

The threshold score of 0.18, determined using the Kolmogorov-Smirnov plot, is used to differentiate between good and bad loans. The model demonstrates a high recall of 0.85 but a relatively low precision of 0.41. This means the model successfully identifies 85% of bad loans, missing only 15%. However, only 41% of the loans predicted as bad are truly bad, while 59% are actually good loans, leading to missed lending opportunities for the company. The model applies a stringent classification approach, making it less likely to classify loans as good.


| Metrics | Training | Testing | Validation |
| ------ | ------ | ------ | ------ |
| AUC ROC | 0.951 | 0.916 | 0.883 |
| Precision | 0.66 | 0.41 |  |
| Recall | 0.84 | 0.85 |  |
| F1-Score | 0.74 | 0.55 |  |
| Accuracy | 0.92 | 0.80 |  |

![alt text](images/roc_auc.png)

### Feature Importances

The most important feature contribution to the model performance is loan_percentage_income, followed by home_mortgage and loan_int_rate.

![alt text](images/feature_importances.png)

## 4. Insights

How each feature contributes to predicting the loan_status is described by the optimal binning result.

- Loan_percent_income or the percentage of the loan to income amount is the strongest feature according to the optimal binning and feature importance. 
The higher loan_percent_income, the higher the risk is. Loan_percent_income of 31% or more has a bad loan probability of 71%.

- When the loan interest rate starts from 14.37%, the bad loan rate is at least 46%.

- Borrowers with income lower than 34,994 have a bad loan rate of 35%.

- There is only 1% of bad loans from the borrowers who own their homes.

- If a borrower has a default history, his loan is 2.5 times more likely to be a bad loan.

View the Kaggle notebook [here](https://www.kaggle.com/code/rendyk/loan-approval)
