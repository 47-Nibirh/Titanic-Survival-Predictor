import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Select important features
df = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # male=1, female=0

# Split data
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "titanic_model.pkl")

print("Model trained and saved successfully!")
