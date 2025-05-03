#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pyplot as plt

# ========= 1. Load Abalone Dataset from Local CSV =========
file_path = r"C:\Users\91989\Downloads\archiveabalone\abalone.csv"
df = pd.read_csv(file_path)

# Print actual column names to verify
print("Columns in dataset:", df.columns.tolist())

# ========= 2. Feature Setup with Correct Column Names =========
features = ['Sex', 'Length', 'Diameter', 'Height',
            'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']

# One-hot encode 'Sex' column
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(), ['Sex'])
], remainder='passthrough')




# ========= (a) Predict Number of Rings - Regression =========
print("\n--- Regression: Predict the number of rings either as a continuous value or as a classification problem.")
X = df[features]
y = df['Rings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

reg_pipeline.fit(X_train, y_train)
y_pred_rings = reg_pipeline.predict(X_test)

print("Mean Squared Error (Rings Prediction):", mean_squared_error(y_test, y_pred_rings))





# ========= (a) Predict Age Group - Classification =========
print("\n--- Classification: Predict the age of abalone from physical measurements using linear regression ")
df['AgeGroup'] = pd.cut(df['Rings'], bins=[0, 9, 12, 30], labels=['Young', 'Middle-aged', 'Old'])

X_cls = df[features]
y_cls = df['AgeGroup']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, stratify=y_cls, test_size=0.2, random_state=42)

cls_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

cls_pipeline.fit(X_train_cls, y_train_cls)
y_pred_cls = cls_pipeline.predict(X_test_cls)


print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))



# ========= (b) Predict Age (Rings + 1.5) - Regression =========
print("\n--- Regression: Predict Age (Rings + 1.5) ---")
df['Age'] = df['Rings'] + 1.5

X_age = df[features]
y_age = df['Age']

X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(
    X_age, y_age, test_size=0.2, random_state=42)

age_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

age_pipeline.fit(X_train_age, y_train_age)
y_pred_age = age_pipeline.predict(X_test_age)

print("Mean Squared Error (Age Prediction):", mean_squared_error(y_test_age, y_pred_age))




# ========= Plot: Actual vs Predicted Age =========
plt.figure(figsize=(6, 4))
plt.scatter(y_test_age, y_pred_age, alpha=0.5, color='teal')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs Predicted Age')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




