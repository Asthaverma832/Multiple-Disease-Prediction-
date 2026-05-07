import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("parkinson's dataset.csv")

# Features and target
X = df.drop(['status', 'name'], axis=1)
y = df['status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print(f"Parkinsons Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save model
pickle.dump(model, open('parkinsons_model.pkl', 'wb'))
pickle.dump(scaler, open('parkinsons_scaler.pkl', 'wb'))
print("Model saved successfully!")