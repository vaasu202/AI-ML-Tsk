````markdown
# Credit Card Fraud Detection

## Overview
This project implements machine learning models to detect fraudulent credit card transactions. The dataset used is the `creditcard.csv`, which contains transaction details along with labels indicating fraud or non-fraud transactions. 

## Libraries Needed
Ensure you have the following libraries installed before running the notebook:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.linear_model import LogisticRegression
import xgboost as xgb_lib
from xgboost import XGBClassifier
```

You can install missing libraries using:

```sh
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn tensorflow xgboost
```

## Dataset
The dataset `creditcard.csv` should be placed in the same directory as this notebook. The dataset is loaded as follows:

```python
df = pd.read_csv("creditcard.csv")
```

### Dataset Information
- The dataset is loaded and basic statistics are displayed.
- The class distribution is printed to highlight the imbalance between fraud and non-fraud transactions.
- Missing values are checked.
- Feature distributions for `Amount` and `Time` are visualized.

## Data Preprocessing
### Feature Scaling
- The `Amount` and `Time` columns are scaled using `StandardScaler`.
- Optionally, SMOTE (Synthetic Minority Over-sampling Technique) can be applied to balance the dataset.
- The dataset is split into training and testing sets with `train_test_split`.

## Models Implemented
### 1. Logistic Regression & XGBoost Classifier
These models are trained using:
- Logistic Regression (`LogisticRegression` from `sklearn`)
- XGBoost Classifier (`XGBClassifier` from `xgboost`)

After training, the following evaluations are performed:
- Classification Report
- ROC-AUC Score
- Confusion Matrix
- Feature Importance (for XGBoost)

To start these models, use:

```python
start_base_and_XGB(df, apply_smote=True)  # Set apply_smote=False to disable SMOTE
```

### 2. Autoencoder Model
- The Autoencoder is trained using only normal transactions.
- It reconstructs transactions, and Mean Squared Error (MSE) is used as an anomaly score.
- A threshold (95th percentile of reconstruction errors) is set to classify transactions as fraudulent or not.
- Evaluation is performed using:
  - Classification Report
  - ROC-AUC Score
  - ROC Curve Visualization

To start the Autoencoder model, use:

```python
start_auto(df, apply_smote=True)  # Set apply_smote=False to disable SMOTE
```

## Visualization
### Receiver Operating Characteristic (ROC) Curve
A function `show_ROC` is defined to plot the ROC curve for model evaluation.

```python
def show_ROC(y_test, next_param):
    fpr, tpr, _ = roc_curve(y_test, next_param)
    auc_score = roc_auc_score(y_test, next_param)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.show()
```

## How to Run
```sh
1. Download the dataset `creditcard.csv` and place it in the same directory as the notebook.
2. Install required dependencies (`pip install -r requirements.txt` if using a requirements file).
3. Open the notebook and run all cells sequentially.
4. Modify function calls to enable/disable SMOTE as needed.
```

## Results
```sh
The output includes:
- Model performance metrics (Precision, Recall, F1-score, AUC-ROC)
- Confusion matrices
- ROC curve visualizations
- Feature importance for XGBoost
```

## Author
```sh
This project was implemented as part of an initiative to detect fraudulent transactions using machine learning models.
```

## License
```sh
This project is licensed under the MIT License.
```
````

