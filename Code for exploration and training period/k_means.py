import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np

df = pd.read_csv('data//clean//order_klav_merge_customerLevel.csv')

#print(df.head())

#Starting to look into correlated values as Silhouette score was very bad meaning a lot of noise and clusters dense areas.
numeric_df = df.select_dtypes(include=[np.number])
# 1- Highly correlated pairs NB Orders / Amount Orders / NB Items. 2- Avg Amount Orders / Max Amount Orders. 3- click / open / accepts marketing / Email Marketing consent
# steps to be taken 

# ########################### Order Intensity ##############################################
#1st step Creating order intensity representing to capture bith value and recency in one score
# # Avoid division by zero or NaNs
# df["DaysSinceRecentOrder"] = df["DaysSinceRecentOrder"].replace(0, 1)
# # Create Order Intensity
# df["Order Intensity"] = df["Amount Orders"] / df["DaysSinceRecentOrder"]
# # Optional: log-transform to reduce skew
# df["Order Intensity"] = np.log1p(df["Order Intensity"])

# ########################## Spend variability #############################################
# # Avoid division by zero
# df["Avg Amount Orders"] = df["Avg Amount Orders"].replace(0, 1)
# # Create Spending Variability
# df["Spending Variability"] = df["Max Amount Orders"] / df["Avg Amount Orders"]
# # Optional: log-transform to normalize
# df["Spending Variability"] = np.log1p(df["Spending Variability"])

##################################### Main Payment method #################################
# # Detect all one-hot encoded payment method columns
# paymeth_cols = [col for col in df.columns if col.startswith('PayMeth_')]

# # Get the payment method name (column name) with the highest value (i.e., the one that’s 1)
# df["Main_Payment_Method"] = df[paymeth_cols].idxmax(axis=1).str.replace("PayMeth_", "")

# # Optional: convert to categorical code for clustering
# df["Main_Payment_Method"] = df["Main_Payment_Method"].astype("category").cat.codes

# Drop highly correlated volume features
#df.drop(columns=["Nb Orders", "Amount Orders", "Nb items"], inplace=True)

# Keep one of first correlated group 

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

# #replace initial source with other + pick top 10
# df['Initial Source New'] = df['Initial Source New'].apply(lambda x: x if x in top_cities.index else 'Other')
# #encode dummy variables
# df = pd.get_dummies(df, columns=['Initial Source New'], drop_first=True)

#replace Last source with other + pick top 10
# df['Last Source New'] = df['Last Source New'].apply(lambda x: x if x in top_cities.index else 'Other')
# #encode dummy variables
# df = pd.get_dummies(df, columns=['Last Source New'], drop_first=True)


# # print(df.head())
# # #Drop columns not used //belong to other models
# #df = df.drop(columns=['Initial Source New', 'CLV', 'Email', 'List SKU', 'Nb Orders', 'Amount Orders', 'Nb items', 'Avg Amount Orders', 'Max Amount Orders', 'click', 'open', 'Accepts Marketing', 'Email Marketing Consent', 'PayMeth_Bancontact', 'PayMeth_Card', 'PayMeth_Ideal', 'PayMeth_Klarna', 'PayMeth_Other', 'PayMeth_Pay Later', 'PayMeth_shopify payments'])
df = df.drop(columns=['CLV', 'Email', 'List SKU', 'Last Source New', 'Initial Source New', 'Days since Date Added'])
#print(df.columns.tolist())
df.to_csv('data//clean//customerLevel_kmeans.csv')
#print(df.dtypes)

# Compute correlation matrix
# corr_matrix = df.corr()
# # Plot heatmap
# plt.figure(figsize=(26, 40))
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, linewidths=0.5)
# plt.title("Correlation Heatmap of Customer Features")
# plt.show()

df = df.drop(columns=['Max Amount Orders', 'Max item amount', 'Days since Profile Created On'])
# corr_matrix = df.corr()
# # Plot heatmap
# plt.figure(figsize=(26, 40))
# sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, linewidths=0.5)
# plt.title("Correlation Heatmap of Customer Features")
# plt.show()

df = df.drop(columns=['PayMeth_Card','PayMeth_Other', 'Always Free Shipping', 'Never Free Shipping'])

#prepare X
X = df.values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# print(df.columns.to_list)

pca = PCA(n_components=5) 
X_reduced = pca.fit_transform(x_scaled)

# inertia = []
# K_range = range(1, 20)

# for k in K_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X)
#     inertia.append(kmeans.inertia_)

# plt.plot(K_range, inertia, marker='o')
# plt.xlabel('Number of clusters (K)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal K')
# plt.grid(True)
# plt.show()

# #K = 4 (safe bet)
# #K = 5 (if you want to be a little more detailed)

# #train model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_reduced)

# #plot results of K-means 
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
# plt.title("K-Means Clustering with PCA (K=4)")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.grid(True)
# plt.show()


# #label results
df['cluster'] = kmeans_labels
df.to_csv('Github//BigDataAI_Project//Train_models//k_means//clustered_data.csv', index=False)

df['cluster'].value_counts().sort_index()

cluster_summary = df.groupby("cluster")[[
    'Avg Amount Orders', 'Nb items', 'Avg Nb items', 'Avg item amount',
    'PayMeth_Bancontact', 'PayMeth_Ideal', 'PayMeth_Klarna',
    'PayMeth_Pay Later', 'PayMeth_shopify payments',
    'Always Discount', 'Never Discount', 'Max Discount Percentage',
    'Same SKU more than once',
    'Email Marketing Consent', 'Accepts Marketing', 'click', 'open',
    'Days since First Active', 'Days since Last Active',
    'Recent City_amsterdam', 'Recent City_den haag', 'Recent City_haarlem',
    'Recent City_rotterdam', 'Recent City_utrecht',
    'Recent Country_be', 'Recent Country_de', 'Recent Country_nl'
]].mean().round(2)

cluster_summary.to_csv("Github\BigDataAI_Project\Train_models\k_means\cluster_summary_profiles.csv")


#analyse data 
cluster_summary.sort_values("Avg Amount Orders", ascending=False)

cluster_labels = {
    0: "VIP / Loyal Customers",
    1: "Dormant Shoppers",
    2: "Occasional Buyers",
    3: "Promo-Engaged Buyers"
}

df['cluster_label'] = df['cluster'].map(cluster_labels)


cluster_summary.head()


#Plot features 
cluster_summary = cluster_summary.reset_index()
cluster_summary["Cluster Label"] = cluster_summary["cluster"].map(cluster_labels)
sns.set(style="whitegrid")
key_metrics = [
    'Avg Amount Orders', 'Nb items', 'Avg Nb items', 'Avg item amount',
    'PayMeth_Bancontact', 'PayMeth_Ideal', 'PayMeth_Klarna',
    'PayMeth_Pay Later', 'PayMeth_shopify payments',
    'Always Discount', 'Never Discount', 'Max Discount Percentage',
    'Same SKU more than once',
    'Email Marketing Consent', 'Accepts Marketing', 'click', 'open',
    'Days since First Active', 'Days since Last Active',
    'Recent City_amsterdam', 'Recent City_den haag', 'Recent City_haarlem',
    'Recent City_rotterdam', 'Recent City_utrecht',
    'Recent Country_be', 'Recent Country_de', 'Recent Country_nl'
]

for feature in key_metrics:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=cluster_summary, x="Cluster Label", y=feature, palette="Set2")
    plt.title(f"{feature} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()
#plot xlusters centers 
centers = pca.transform(kmeans.cluster_centers_)

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
# plt.title("K-Means Clustering with Centroids (PCA)")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend()
# plt.grid(True)
# plt.show()

# #explore clusters charecterestic
# cluster_summary = df.groupby('cluster').mean()
# print(cluster_summary)

# print(df['cluster'].value_counts())


# # # Exploring clusters for business overview
# # sns.boxplot(x='cluster', y='DaysSinceRecentOrder', data=df)
# # plt.title("DaysSinceRecentOrder")
# # plt.show()

# cluster_profiles = df.groupby("cluster").mean(numeric_only=True)

# # Count of records in each cluster
# cluster_counts = df["cluster"].value_counts().sort_index()

# # tools.display_dataframe_to_user(name="Cluster Profiles", dataframe=cluster_profiles)
# print(cluster_profiles)

# print(cluster_counts)

# cluster_profiles_with_counts = cluster_profiles.copy()
# cluster_profiles_with_counts['count'] = cluster_counts

# # Save to CSV
# summary_output_path = "\cluster_summary_profiles.csv"
# cluster_profiles_with_counts.to_csv(summary_output_path)

# #Cities per cluster
# city_columns = [col for col in df.columns if col.startswith('Recent City_')]
# city_distribution = df.groupby("cluster")[city_columns].mean()

# city_distribution *= 100

# plt.figure(figsize=(12, 6))
# sns.heatmap(city_distribution, annot=True, fmt=".1f", cmap="Blues")
# plt.title("City Distribution by Cluster (%)")
# plt.xlabel("City")
# plt.ylabel("Cluster")
# plt.tight_layout()
# plt.show()


# cluster_labels = {
#     0: "Dormant Shoppers",
#     1: "VIP / Loyal Customers",
#     2: "Occasional Buyers",
#     3: "Inactive Customers"
# }

# df['cluster_label'] = df['cluster'].map(cluster_labels)

# Calculate Silhouette Score
sil_score = silhouette_score(X_reduced, kmeans_labels)
print(f"K-Means Silhouette Score: {sil_score:.2f}")

# # # #loop sil score for best k 
# sil_scores = []

# for k in range(2, 9):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X_reduced)
#     score = silhouette_score(X_reduced, labels)
#     sil_scores.append((k, score))
#     print(f"k={k}, silhouette score={score:.4f}")

# ks, scores = zip(*sil_scores)
# plt.plot(ks, scores, marker='o')
# plt.title("Silhouette Scores for Different K")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.grid(True)
# plt.show()

# # Days Since Recent Order
# # plt.figure(figsize=(10, 5))
# # sns.boxplot(x='cluster_label', y='DaysSinceRecentOrder', data=df)
# # plt.title("Days Since Recent Order by K-Means Cluster")
# # plt.ylabel("Days")
# # plt.xticks(rotation=20)
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # Total Amount Spent
# # plt.figure(figsize=(10, 5))
# # sns.boxplot(x='cluster_label', y='Amount Orders', data=df)
# # plt.title("Total Amount Spent by K-Means Cluster")
# # plt.ylabel("Amount (€)")
# # plt.xticks(rotation=20)
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # Average Order Value
# # plt.figure(figsize=(10, 5))
# # sns.boxplot(x='cluster_label', y='Avg Amount Orders', data=df)
# # plt.title("Average Order Value by K-Means Cluster")
# # plt.ylabel("Avg Order (€)")
# # plt.xticks(rotation=20)
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()


# # Cluster 0 
# # number of users = 4519
# # Feature	Value
# # DaysSinceRecentOrder	~260
# # Nb Orders	 ~2.5
# # Amount Orders	 ~150
# # Avg Amount Orders	 ~57
# # Email Click/Open	 Low
# # DaysSinceLastActive	 High (~250+)

# #Interpretation:

# #Infrequent orders, low activity

# #Still spending, but not recently
# #Label: Dormant Shoppers or Low-Engagement Buyers


#Cluster 1 (410 users — small group)
#Feature	             Value
#DaysSinceRecentOrder	 ~190
#Nb Orders	             High (~10)
#Amount Orders	         Very High (~700)
#Avg Amount Orders	     ~70
#Email Click/Open	     High
#DaysSinceLastActive	 ~180

#Interpretation:
#Small but valuable segment
#High engagement, high spend
#Label: VIP / Loyal Customers


#Cluster 2 (5,142 users)
#Feature	            Value
#DaysSinceRecentOrder	~210
#Nb Orders	            ~2.5
#Amount Orders	        ~140
#Avg Amount Orders	    ~55
#Email activity	        Moderate
#Discount usage	        Mixed

# Interpretation:
# Similar to Cluster 0 but a bit more recent and active  
# Label: Occasional Buyers or Casual Customers

#Cluster 3 (7,664 users — largest group)
#Feature	            Value
#DaysSinceRecentOrder	~385 (very high)
#Nb Orders	            ~1.2
#Amount Orders	        Low (~40)
#Avg item amount	    Low
#Marketing interaction	Very low
#DaysSinceLastActive	~380

#Interpretation:
#Likely churned, inactive
#Label: Inactive Customers