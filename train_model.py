import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Convert all CSVs in data/ to npy files
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        label = file.split("_")[0]  # e.g., "A" from "A_hand_landmarks.csv"
        df = pd.read_csv(os.path.join(data_dir, file))
        data = df.drop('label', axis=1).values
        np.save(os.path.join(data_dir, f"{label}.npy"), data)

# Prepare data for training
X, y = [], []

for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        label = file.split(".")[0]
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        X.extend(data)
        y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
joblib.dump(model, 'model.pkl')
