import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

# loading and reading dataset
df = pd.read_csv("Mall_Customers.csv")

# Select relevant features
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data with StandardScaler from the scikit-learn. This will scale the data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


