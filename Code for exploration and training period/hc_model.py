import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("data//clean//order_klav_merge_customerLevel.csv")

# Drop irrelevant or redundant columns
df = df.drop(columns=[
    'CLV', 'Email', 'List SKU', 'Last Source New', 'Initial Source New',
    'Days since Date Added'
], errors='ignore')

# One-hot encode top cities and countries
top_cities = df['Recent City'].value_counts().nlargest(5).index
top_countries = df['Recent Country'].value_counts().nlargest(3).index

df['Recent City'] = df['Recent City'].where(df['Recent City'].isin(top_cities), other='Other')
df['Recent Country'] = df['Recent Country'].where(df['Recent Country'].isin(top_countries), other='Other')

df = pd.get_dummies(df, columns=['Recent City', 'Recent Country'], drop_first=False)

# Prepare for HC: scale + PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

X_scaled = StandardScaler().fit_transform(df)
X_reduced = PCA(n_components=0.95).fit_transform(X_scaled)

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=5, metric='manhattan', linkage='average')
df['hc_cluster'] = hc.fit_predict(X_reduced)

# Create updated cluster summary
hc_cluster_summary = df.groupby("hc_cluster").mean(numeric_only=True).reset_index()

# Apply business-friendly labels
label_map = {
    0: "Dormant Shoppers",
    1: "Churned Casual Buyers",
    2: "New & Engaged Buyers",
    3: "Loyal Core Customers",
    4: "Promo-Responsive Customers"
}
hc_cluster_summary['Cluster Label'] = hc_cluster_summary['hc_cluster'].map(label_map)

# Plot updated cities and countries by cluster
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Define updated column names
city_cols = [
    'Recent City_amsterdam', 'Recent City_den haag',
    'Recent City_haarlem', 'Recent City_rotterdam', 'Recent City_utrecht'
]

country_cols = ['Recent Country_be', 'Recent Country_de', 'Recent Country_nl']

# Plot city distributions
for city in city_cols:
    if city in hc_cluster_summary.columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=hc_cluster_summary, x='Cluster Label', y=city, palette="pastel")
        plt.title(f"{city.replace('_', ' ').title()} by Cluster")
        plt.xlabel("Customer Segment")
        plt.ylabel("City Proportion")
        plt.tight_layout()
        plt.show()

# Plot country distributions
for country in country_cols:
    if country in hc_cluster_summary.columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=hc_cluster_summary, x='Cluster Label', y=country, palette="pastel")
        plt.title(f"{country.replace('_', ' ').title()} by Cluster")
        plt.xlabel("Customer Segment")
        plt.ylabel("Country Proportion")
        plt.tight_layout()
        plt.show()


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.metrics import silhouette_score


# df = pd.read_csv('data//clean//order_klav_merge_customerLevel.csv')

# #change cities to order catergorical data
# city_counts = df['Recent City'].value_counts().sort_values(ascending=True)
# city_order = {city: i for i, city in enumerate(city_counts.index)}
# df['city_ordered'] = df['Recent City'].map(city_order)

# #countries 
# country_counts = df['Recent Country'].value_counts().sort_values(ascending=True)
# country_order = {country: i for i, country in enumerate(country_counts.index)}
# df['country_ordered'] = df['Recent Country'].map(country_order)

# df['city_ordered'] = df['Recent City'].map(city_order)
# df['country_ordered'] = df['Recent Country'].map(country_order)

# top_cities = df['Recent City'].value_counts().nlargest(5).index
# top_countries = df['Recent Country'].value_counts().nlargest(3).index

# df['Recent City'] = df['Recent City'].where(df['Recent City'].isin(top_cities), other='Other')
# df['Recent Country'] = df['Recent Country'].where(df['Recent Country'].isin(top_countries), other='Other')

# df = pd.get_dummies(df, columns=['Recent City', 'Recent Country'])

# #df = df.drop(columns=['CLV', 'Email', 'List SKU', 'Last Source New', 'Initial Source New', 'Days since Date Added', 'Recent City', 'Recent Country'])
# df = df.drop(columns=['CLV', 'Email', 'List SKU', 'Last Source New', 'Initial Source New', 'Days since Date Added'])

# X = df.values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=0.95)
# X_pca = pca.fit_transform(X_scaled)

# linked = linkage(X_pca, method='average')
# model = AgglomerativeClustering(n_clusters=5, metric='manhattan', linkage='average')
# hc_labels = model.fit_predict(X_pca)

# df['hc_cluster'] = hc_labels

# hc_cluster_summary = df.groupby('hc_cluster').mean(numeric_only=True).round(2)
# hc_cluster_summary = hc_cluster_summary.reset_index()

# hc_cluster_summary.to_csv("Github\BigDataAI_Project\Train_models\hc\summary_hc.csv")

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Let's simulate an example barplot with the available data structure for one feature
# # First, relabel clusters with interpretable names if needed
# # v = {
# #     0: "Cluster 0",
# #     1: "Cluster 1",
# #     2: "Cluster 2",
# #     3: "Cluster 3",
# #     4: "Cluster 4"
# # }
# #hc_cluster_summary['Cluster Label'] = hc_cluster_summary['hc_cluster'].map(cluster_labels_map)

# cluster_labels = {
#     0: "Dormant Shoppers",
#     1: "Churned Casual Buyers",
#     2: "New & Engaged Buyers",
#     3: "Loyal Core Customers",
#     4: "Promo-Responsive Customers"
# }
# hc_cluster_summary['Cluster Label'] = hc_cluster_summary['hc_cluster'].map(cluster_labels)

# key_metrics = [
#     'Avg Amount Orders', 'Nb items', 'Avg Nb items', 'Avg item amount',
#     'PayMeth_Bancontact', 'PayMeth_Ideal', 'PayMeth_Klarna',
#     'PayMeth_Pay Later', 'PayMeth_shopify payments',
#     'Always Discount', 'Never Discount', 'Max Discount Percentage',
#     'Same SKU more than once',
#     'Email Marketing Consent', 'Accepts Marketing', 'click', 'open',
#     'Days since First Active', 'Days since Last Active',
#     'Recent City_amsterdam', 'Recent City_den haag', 'Recent City_haarlem',
#     'Recent City_rotterdam', 'Recent City_utrecht',
#     'Recent Country_be', 'Recent Country_de', 'Recent Country_nl'
# ]

# # for feature in key_metrics:
# #     if feature in hc_cluster_summary.columns:
# #         plt.figure(figsize=(10, 6))
# #         sns.barplot(data=hc_cluster_summary, x="Cluster Label", y=feature, palette="Set2")
# #         plt.title(f"{feature} by Cluster")
# #         plt.xlabel("Cluster")
# #         plt.ylabel(feature)
# #         plt.tight_layout()
# #         plt.show()


# country_cols = ['Recent Country_be', 'Recent Country_de', 'Recent Country_nl']
# sns.set(style="whitegrid")
# for country in country_cols:
#     if country in hc_cluster_summary.columns:
#         plt.figure(figsize=(8, 4))
#         sns.barplot(data=hc_cluster_summary, x='Cluster Label', y=country, palette="pastel")
#         plt.title(f"{country.replace('_', ' ').title()} by Cluster")
#         plt.xlabel("Customer Segment")
#         plt.ylabel("Country Proportion")
#         plt.tight_layout()
#         plt.show()


# city_cols = [
#     'Recent City_amsterdam', 'Recent City_den haag', 'Recent City_haarlem',
#     'Recent City_rotterdam', 'Recent City_utrecht'
# ]

# # Plot each city by cluster label
# for city in city_cols:
#     if city in hc_cluster_summary.columns:
#         plt.figure(figsize=(8, 4))
#         sns.barplot(data=hc_cluster_summary, x='Cluster Label', y=city, palette="pastel")
#         plt.title(f"{city.replace('_', ' ').title()} by Cluster")
#         plt.xlabel("Customer Segment")
#         plt.ylabel("City Proportion")
#         plt.tight_layout()
#         plt.show()

# # plt.figure(figsize=(10, 5))
# # dendrogram(linked)
# # plt.title('Hierarchical Clustering Dendrogram')
# # plt.xlabel('Sample Index')
# # plt.ylabel('Distance')
# # plt.show()

# #model = AgglomerativeClustering(n_clusters=4, linkage='ward')

# score = silhouette_score(X_pca, hc_labels, metric='manhattan')
# print(f"Silhouette Score (Manhattan): {score:.3f}")