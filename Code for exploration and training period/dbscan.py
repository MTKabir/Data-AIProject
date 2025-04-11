import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import hdbscan


df = pd.read_csv('data//clean//order_klav_merge_customerLevel.csv')

#print(df.head())

#get most 10 cities 
top_cities = df['Recent City'].value_counts().head(5)
top_countries = df['Recent Country'].value_counts().head(3)
#print(top_cities)

#replace other cities with Other, 
df['Recent City'] = df['Recent City'].apply(lambda x: x if x in top_cities.index else 'Other')
#encode dummy variables
df = pd.get_dummies(df, columns=['Recent City'], drop_first=True)

#replace countries with other + pick top 10 countries 
df['Recent Country'] = df['Recent Country'].apply(lambda x: x if x in top_countries.index else 'Other')
#encode dummy variables
df = pd.get_dummies(df, columns=['Recent Country'], drop_first=True)

# print(df.head())
# #Drop columns not used //belong to other models
df = df.drop(columns=['CLV', 'Email', 'List SKU', 'Last Source New', 'Initial Source New', 'Days since Date Added'])
df = df.drop(columns=['Max Amount Orders', 'Max item amount', 'Days since Profile Created On'])
df = df.drop(columns=['PayMeth_Card','PayMeth_Other', 'Always Free Shipping', 'Never Free Shipping'])
#print(df.dtypes)

X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#reduce dimensions 
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=40)
dbscan_labels = dbscan.fit_predict(X_reduced)

df['dbscan_cluster'] = dbscan_labels


#analysis 
cluster_summary = df.groupby('dbscan_cluster').mean(numeric_only=True).round(2)
cluster_sizes = df['dbscan_cluster'].value_counts().sort_index()
cluster_summary['Cluster Size'] = cluster_sizes

cluster_sizes = df['dbscan_cluster'].value_counts().sort_index()
cluster_summary['Cluster Size'] = cluster_sizes

cluster_summary.to_csv("Github\BigDataAI_Project\Train_models\DBSCAN\DBSCAN_Cluster_Summary.csv")

cluster_summary = cluster_summary.reset_index()
#sort and select key columns 
sorted_clusters = cluster_summary[cluster_summary['dbscan_cluster'] != -1].sort_values(by="Cluster Size", ascending=False)

# Select key behavioral columns for comparison
selected_columns = [
    'dbscan_cluster', 'Cluster Size', 'Nb Orders', 'Amount Orders', 'Avg Amount Orders',
    'DaysSinceRecentOrder', 'Days since Last Active', 'click', 'open', 'Accepts Marketing'
]

sorted_clusters_view = sorted_clusters[selected_columns]

comparison_view = sorted_clusters[selected_columns].sort_values(by="Cluster Size", ascending=False)
comparison_view.reset_index(drop=True, inplace=True)
#view
comparison_view.to_csv("Github\BigDataAI_Project\Train_models\DBSCAN\comparison_with_noise_cluster.csv", index=False)
#################################################
unique, counts = np.unique(dbscan_labels, return_counts=True)
print(dict(zip(unique, counts)))

valid = dbscan_labels != -1
if np.any(valid):
    sil = silhouette_score(X_reduced[valid], dbscan_labels[valid])
    print(f"DBSCAN Silhouette Score (no noise): {sil:.4f}")
else:
    print("No valid clusters for silhouette score.")

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=dbscan_labels, cmap='tab10', s=10)
plt.title("DBSCAN Clustering (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

clusterer = hdbscan.HDBSCAN(min_cluster_size=350, min_samples=5, prediction_data=True)
hdb_labels = clusterer.fit_predict(X_reduced)

unique, counts = np.unique(hdb_labels, return_counts=True)
print("HDBSCAN cluster distribution:", dict(zip(unique, counts)))

# Silhouette (exclude noise)
valid = hdb_labels != -1
if np.any(valid):
    sil_score = silhouette_score(X_reduced[valid], hdb_labels[valid])
    print(f"HDBSCAN Silhouette Score (excluding noise): {sil_score:.4f}")
else:
    print("No valid clusters to evaluate.")

# df['dbscan_cluster'] = dbscan_labels
# cluster_summary = df.groupby('dbscan_cluster').mean(numeric_only=True)
# big_clusters = df['dbscan_cluster'].value_counts()
# big_clusters = big_clusters[big_clusters > 300].index
# df_big = df[df['dbscan_cluster'].isin(big_clusters)]

# df_big.groupby('dbscan_cluster').mean(numeric_only=True)

# output_path = "Github//BigDataAI_Project//Train_models//DBSCAN//dbscan_clustered_data.csv"
# df.to_csv(output_path, index=False)

# output_path

#show analysis on largest clusters 

# top_clusters = [0, 4, 7, 6, 16]

# Filter the original DataFrame to only those clusters
# df_top_clusters = df[df['dbscan_cluster'].isin(top_clusters)].copy()

# # Map human-friendly labels
# cluster_labels = {
#     0: "Churned Single-Order",
#     4: "Lapsed Valuable Buyers",
#     7: "Lapsed Casual Buyers",
#     6: "Newer One-Timers",
#     16: "Potential Repeaters"
# }
# df_top_clusters['cluster_label'] = df_top_clusters['dbscan_cluster'].map(cluster_labels)

# # Plot comparison of DaysSinceRecentOrder
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='cluster_label', y='DaysSinceRecentOrder', data=df_top_clusters)
# plt.title("Days Since Recent Order by Cluster")
# plt.ylabel("Days")
# plt.xticks(rotation=20)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Plot comparison of Amount Orders
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='cluster_label', y='Amount Orders', data=df_top_clusters)
# plt.title("Total Amount Spent by Cluster")
# plt.ylabel("Amount (€)")
# plt.xticks(rotation=20)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Plot comparison of Avg Amount Orders
# plt.figure(figsize=(10, 5))
# sns.boxplot(x='cluster_label', y='Avg Amount Orders', data=df_top_clusters)
# plt.title("Average Order Value by Cluster")
# plt.ylabel("Avg Order (€)")
# plt.xticks(rotation=20)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#explain initial attempt 
#eps is too small more that 80% of the data are ourliers 
#groups are tightly packed, which is not ideal like identifies anomalies 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#raising eps now to 1.0
#59% of data is detected as outliers + a lot of tiny clusters
#DBSCAN still tightly clustering
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#raising eps now to 1.5 + reduce dimentions before DBSCAN in order to reduce noise 
#Much better results, as DBSCAN now starts to form some dense clusters 
#outliers are 40% of the data 
#Clusters formed:
#175 clusters

#large groups:
#Cluster 6: 1,078
#Cluster 0: 992
#Cluster 3: 974
#Cluster 5: 701
#Cluster 17: 525
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#esp=2.0 reduced dimensions via PCA show much better results 
#noise 25% looks more like a healthy level than previous practice
#clusters formed 
# Total: 210 clusters

# Of which:
# Several large clusters:
# Cluster 0: 1,092
# Cluster 4: 1,079
# Cluster 7: 1,079
# Cluster 6: 769
# Cluster 10: 509

# Many more in the 300–500 range
# Plenty of medium clusters (~50–200 points)
# Some small clusters (5–30 points), which is okay if they’re meaningful
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Interpeting Large clusters 

# Cluster 0 (1092 users)
# DaysSinceRecentOrder: 386 → almost all made their last order over a year ago
# Orders: ~1.05 → only 1 order
# Low basket size, low engagement
# Likely: "Dormant One-Timers"
# Suggested Label: Churned Single-Order Users

#####################################
# Cluster 4 (1079 users)
# Similar to Cluster 0, slightly higher Avg Amount Orders (~70) and items
# Also inactive, but spent more
# Likely: "Past High Spenders"
# Suggested Label: Lapsed Valuable Buyers
######################################
# Cluster 7 (1079 users)
# Also inactive (DaysSinceRecentOrder ~386)
# Lowest total spend and order value
# Probably low-engagement, low-value customers
# Likely: "Low-Value, Inactive"
# Suggested Label: Lapsed Casual Buyers
######################################
# Cluster 6 (769 users)
# More recent activity (DaysSinceRecentOrder ~178)
# Decent spend, basket value, item count
# Still light on frequency (1 order), but not churned
# Likely: "Recent One-Time Buyers"
# Suggested Label: Newer One-Timers
######################################
# Cluster 16 (654 users)
# Moderate recency (~218)
# Slightly higher spend and average order value
# Very similar to Cluster 6, but slightly more value-driven
# Likely: "Mid-Recency, One-Time Buyers"
# Suggested Label: Potential Repeaters
#####################################
# Labels summary
# Cluster	Count	Suggested Label
# 0	         1092	Churned Single-Order Users
# 4	         1079	Lapsed Valuable Buyers
# 7	         1079	Lapsed Casual Buyers
# 6	         769	Newer One-Timers
# 16	     654	Potential Repeaters