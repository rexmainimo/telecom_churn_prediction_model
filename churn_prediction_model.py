#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Churn Prediction Model using RandomForestClassifier
This script implements a machine learning pipeline for predicting customer churn.
It includes feature engineering, preprocessing, feature selection, and hyperparameter tuning.
The model is evaluated on test and evaluation datasets, optimized for recall to identify churn cases.
Author: Rex Mainimo
Date: June 26, 2025
"""


# Telecom Customer Churn Analysis.   
# Dataset from Kaggle, link: https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")

print("Path to dataset files:", path)


# In[2]:


path = r"C:\Users\maini\.cache\kagglehub\datasets\mnassrib\telecom-churn-datasets\versions\1"

train_df = pd.read_csv(path + "\\churn-bigml-80.csv")
train_df.head()


# In[3]:


train_df.info() 


# In[4]:


train_df.describe() 


# In[5]:


train_df.shape


# In[6]:


train_df.nunique()


# In[8]:


train_df.isnull().sum()


# For a proper Data Pipeline and SQL-focused project, I will load the training and evaluating dataset into MySQL(local) database   
# The dataset can be downloaded from the Kaggle link above.

# In[7]:


# train data
engine = create_engine("mysql+pymysql://root:@localhost/telecom_churn_dataset") 

# train_df.to_sql(name="train_data", con=engine, if_exists="replace", index=False) 
# print("DataFrame written to SQL table successfully.")


# In[8]:


# evaluation data

eval_df = pd.read_csv(path + "\\churn-bigml-20.csv") 
eval_df.head()


# In[9]:


eval_df.info() 
eval_df.describe()


# In[10]:


eval_df.isnull().sum()


# In[16]:


eval_df.to_sql(name="eval_data", con=engine, if_exists="replace", index=False) 
print("Eval DataFrame written to SQL table successfully.")


# Extracting training dataset from the database.

# In[ ]:


df = ""

try:
    connection = engine.connect()
    print("Connection successful, fetching data from SQL table.") 
    
    training_query = "SELECT * FROM train_data" 
    df = pd.read_sql(training_query, con=engine)
    connection.close()
except Exception as e:
    print(f"SQL Connection or query failed due to: {e}")
    print("defaulting to csv file.") 
    df = pd.read_csv("churn-bigml-80.csv")


df.head()


# Exploratory Data Analysis

# In[13]:


df.Churn.value_counts()


# In[15]:


sns.countplot(x="Churn", data=df)

plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

#Churn rate
churn_rate = df.Churn.value_counts(normalize=True) * 100
print(f"Churn Rate:\n {churn_rate.round(2)}%") 


# In[15]:


num_columns = ["Total day minutes", "Total day minutes", "Total night minutes", "Total intl minutes", "Total eve minutes"]
for col in num_columns:
    sns.histplot(data=df, x=col, hue="Churn", kde=True)
    plt.title("Distribution of " + col + " by Churn")
    plt.show()


# In[16]:


churn_rate_by_state = df.groupby("State")["Churn"].mean().sort_values()

ordered_states = churn_rate_by_state.index.tolist()

plt.figure(figsize=(16, 8))
sns.countplot(x="State", hue="Churn", data=df, order=ordered_states) 
plt.title("Distribution of Churn by State")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.show()


# A state like Texas appear to have a higher churn rate than most states.

# In[17]:


cat_columns = ["Area code", "International plan", "Voice mail plan"]
for col in cat_columns:
    sns.countplot(x=col, hue="Churn", data=df)
    plt.title(f"Distribution of Churn by {col}")
    plt.show()


# In[18]:


num_cols = df.select_dtypes(exclude=["object"])
corr = num_cols.corr(numeric_only=True)

plt.figure(figsize=(12, 8)) 
sns.heatmap(corr, annot=True, cmap="coolwarm")


# In[19]:


df.groupby("International plan")["Churn"].mean() 


# Feature Engineering

# In[55]:


counts_series = df["Customer service calls"].value_counts()

for value, count in counts_series.items():
    print(f"Customer service calls: {value}, Count: {count}")   


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

# Transformer for mapping "Yes"/"No" to 1/0
class BinaryMapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["International plan"] = X["International plan"].map({"Yes": 1, "No": 0})
        X["Voice mail plan"] = X["Voice mail plan"].map({"Yes": 1, "No": 0})
        return X

# Transformer for creating total minutes and calls
class TotalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Total minutes"] = X["Total day minutes"] + X["Total eve minutes"] + X["Total night minutes"] + X["Total intl minutes"]
        X["Total calls"] = X["Total day calls"] + X["Total eve calls"] + X["Total night calls"] + X["Total intl calls"]
        return X

# Transformer for creating high customer service calls
class HighServiceCalls(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["High customer service calls"] = (X["Customer service calls"] > 4).astype(int)
        return X


# In[45]:


numeric_cols = [
    "Account length", "Number vmail messages", "Total day minutes",
    "Total day calls", "Total day charge", "Total eve minutes",
    "Total eve calls", "Total eve charge", "Total night minutes",
    "Total night calls", "Total night charge", "Total intl minutes",
    "Total intl calls", "Total intl charge", "Customer service calls",
    "International plan", "Voice mail plan", "Total minutes",
    "Total calls", "High customer service calls"
]


# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

 

# Preprocessor for one-hot encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["State", "Area code"]),
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"
)

feature_selection = Pipeline(steps=[
    ("selector", SelectFromModel(estimator=RandomForestClassifier(class_weight="balanced", random_state=42)))
])

# Full pipeline
pipeline = Pipeline(steps=[
    ("binary_mapper", BinaryMapper()),
    ("total_features", TotalFeatures()),
    ("high_service_calls", HighServiceCalls()),
    ("preprocessor", preprocessor),
    ("feature_selection", feature_selection),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
])


# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split

# Parameter grid Random Forest
param_grid = {
    "feature_selection__selector__estimator__n_estimators": [100, 300],
    "feature_selection__selector__estimator__max_depth": [5, 10],
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

# Data split
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Recall Score (CV):", grid_search.best_score_)


# In[ ]:


from sklearn.metrics import recall_score, classification_report, confusion_matrix 
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score

best_model = grid_search.best_estimator_

#y_pred = best_model.predict(X_test) 
y_proba = best_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# precision vs. recall
import matplotlib.pyplot as plt
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall vs. Threshold")
plt.show()

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5]:
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))



# The primary objective of this churn prediction model is to maximize the capture of True Positives, i.e., customers who are actually at risk of churning. By applying class balancing and model adjustments, we significantly improved the model's ability to correctly identify churners.
# 
# This approach naturally increases the number of False Positives — customers predicted to churn who would have otherwise stayed. However, this trade-off is acceptable in a churn prevention context, as it enables the business to proactively target at-risk customers, ultimately reducing overall churn and protecting revenue.
# 
# In other words, it is better to mistakenly intervene with some loyal customers than to miss the opportunity to retain actual churners.
# 
# The focus then will be on Recall: How good the model is at capturing actual positives(churners) which often lowers Precision!

# In[49]:


threshold = 0.4

y_pred = (y_proba >= threshold).astype(int)
test_recall = recall_score(y_test, y_pred)
print("Test Recall Score:", test_recall)

# Detailed evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[ ]:


# selected features from SelectFromModel
feature_selection_step = best_model.named_steps["feature_selection"].named_steps["selector"]
selected_features_mask = feature_selection_step.get_support()

# feature names after preprocessing
preprocessor = best_model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

# Filter selected feature names
selected_features = feature_names[selected_features_mask]
print("Selected Features:", selected_features)

# feature importances from the classifier
classifier = best_model.named_steps["classifier"]
importances = classifier.feature_importances_
feature_importance = dict(zip(selected_features, importances))
print("\nFeature Importances:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")


# Final Evaluation with Evaluation dataset

# In[ ]:


df_eval = ""

 
try:
    connection = engine.connect()
    eval_query = "SELECT * FROM eval_data"
    print("Connection successful, fetching data from SQL table.")
    df_eval = pd.read_sql(eval_query, con=engine)
    connection.close()
except Exception as e:
    print(f"Connection or query failed, due to {e}") 
    print("defaulting to csv file.") 
    df_eval = pd.read_csv("churn-bigml-20.csv")


df_eval.head()


# In[58]:


X_eval = df_eval.drop("Churn", axis=1) 
y_eval = df_eval["Churn"]


print("Evaluation Dataset dtypes:\n", X_eval.dtypes)
print("\nTarget Distribution:\n", y_eval.value_counts(normalize=True))

# Best Model from GridSearchCV
best_model = grid_search.best_estimator_
print("\nBest Parameters from GridSearchCV:", grid_search.best_params_)

# Predict on Evaluation Dataset
y_proba_eval = best_model.predict_proba(X_eval)[:, 1]
threshold = 0.4 
y_pred_eval = (y_proba_eval >= threshold).astype(int)


# In[59]:


# Model Evaluation
print("\nEvaluation Dataset Results:")
print("Test Recall Score:", recall_score(y_eval, y_pred_eval))
print("\nClassification Report:\n", classification_report(y_eval, y_pred_eval))
print("\nConfusion Matrix:\n", confusion_matrix(y_eval, y_pred_eval))

# Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_eval, y_proba_eval)
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall vs. Threshold (Evaluation Dataset)")
plt.show()


# In[ ]:




