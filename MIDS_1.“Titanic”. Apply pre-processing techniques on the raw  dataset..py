#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
df = pd.read_csv(r"C:\Users\91989\Desktop\AICS\Titanic-Dataset - Titanic-Dataset.csv")

# Display column names to verify structure
print("Columns in dataset:\n", df.columns)

# Drop unwanted columns (if they exist)
for col in ['Cabin', 'PassengerId', 'Name', 'Ticket']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Create 'Family' feature
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['Family'] = df['SibSp'] + df['Parch'] + 1

# Fill missing values
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode 'Sex' to numeric
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Standardize numerical features
scaler = StandardScaler()
for col in ['Age', 'Fare']:
    if col in df.columns:
        df[[col]] = scaler.fit_transform(df[[col]])

# Correlation with 'Survived'
if 'Survived' in df.columns:
    corr = df.corr(numeric_only=True)
    print("\nCorrelation with Survived:\n", corr['Survived'].sort_values(ascending=False))

# Random Forest for Feature Importance
if 'Survived' in df.columns:
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nFeature Importances:\n", importances.sort_values(ascending=False))


# In[ ]:




