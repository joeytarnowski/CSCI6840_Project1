import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def eda(name, data):
    print(f"\n{name} distribution: \n")
    print(f"Median: {data.median()}")
    print(f"Mean: {data.mean()}")
    print(f"Standard Deviation: {data.std()}")
    print(f"Range: {data.max() - data.min()} (from {data.min()} to {data.max()})")

# loading and reading dataset
df = pd.read_csv("Mall_Customers.csv")
print("\nData head: \n")
print(df.head())

# Check for missing values
print(f"\nNull Values: \n{df.isnull().values.any()}")
print()

# Data information 
print("\nData info: \n")
df.info()

# Exploratory Data Analysis
# Gender ratio
print("\nGender ratio: \n")
gender = df['Gender'].value_counts()
print(gender)

# Age distribution
eda("Age", df['Age'])

# Annual income distribution
eda("Annual Income", df['Annual Income (k$)'])

# Spending score distribution
eda("Spending Score", df['Spending Score (1-100)'])

# Data normalization
# Select relevant features
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data with StandardScaler from the scikit-learn. This will scale the data to have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
