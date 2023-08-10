import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Select relevant features (spending scores)
X = data.iloc[:, 3:5].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using the Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
num_clusters = 4

# Initialize the KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Add cluster labels to the original data
data['Cluster'] = kmeans.labels_

# Visualize the clusters
sns.scatterplot(x='Spending Score (1-100)', y='Annual Income (k$)', hue='Cluster', data=data)
plt.title('Customer Segmentation')
plt.show()
