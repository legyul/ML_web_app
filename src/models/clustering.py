# Load libraries
import pandas as pd
from .common import spark, spark_processing, pandas_processing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

matplotlib.use('Agg')

# Functions
# Find the useful variables to cluster
def eliminate_high_correlation(data, threshold=0.8):
    # Eliminate the highly correlated features
    
    if isinstance(data, pd.DataFrame):
        corr_matrix = data.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        return data.drop(columns = to_drop)
    
    elif isinstance(data, spark.sql.dataframe.DataFrame):
        columns = data.columns
        to_drop = set()  # Set the stored the eliminated columns

        # Calculate the correlation between each column
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                corr_value = data.stat.corr(col1, col2)
                if corr_value > threshold:
                    # Add highly correlated columns to the list to be removed
                    to_drop.add(col2)

        data_cleaned = data.drop(*to_drop)

        return data_cleaned

def eliminate_low_variance(data, threshold=0.01):
    # Eliminate the low variacne features
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)
    reduce = selector.fit_transform(data)
    return pd.DataFrame(reduce, columns=data.columns[selector.get_support()])

def apply_pca(data, variance_threshold=0.95, max_component=10):
    pca_info = ""
    # Check the number of variance
    n_features = data.shape[1]
    if n_features < 2:
        pca_info += "Insufficient features after filtering. Returning original data."
        return data, pca_info
    
    # Using PCA calculate the ratio of cumulative variance
    pca = PCA()
    pca.fit(data)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # Determining the minimum number of components based on variance ratio
    n_components_by_variance = (cumulative_variance < variance_threshold).sum() + 1
    n_components = min(n_components_by_variance, max_component, n_features)

    n_components = max(n_components, 2) if n_features > 1 else 1
    pca_info += f"(explained variance threshold: {variance_threshold * 100}%)."

    # Apply PCA
    pca = PCA(n_components= n_components)
    reduced_data = pca.fit_transform(data)
    reduced_data = pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)], index=data.index)

    component_importance = pd.DataFrame(pca.components_, columns=data.columns, index=[f"PC{i+1}" for i in range(n_components)])

    return reduced_data, component_importance, pca_info

def filter_data(data, threshold_corr=0.8, threshold_var=0.01, explained_variance=0.95, max_components=10):
    # Filter the highly correlated features
    data = eliminate_high_correlation(data, threshold=threshold_corr)

    # Filter the low variacne features
    data = eliminate_low_variance(data, threshold=threshold_var)

    # Apply PCA
    data, component_importane, pca_info = apply_pca(data, variance_threshold=explained_variance, max_component=max_components)

    return data, component_importane, pca_info

# Perform PCA for visualization
def visualize_pca(data, mode):
    if mode == 'spark':
        scaled_data = spark_processing.spark_scaled_df(data)
    
    elif mode == 'pandas':
        scaled_data = pandas_processing.pandas_scale_df(data)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    return pca_data

# Determine optimal number of clusters using elbow method
def elbow(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Find the elbow point
    x1, y1 = 1, wcss[0]
    x2, y2 = len(wcss), wcss[-1]
    dis = []
    for i in range(1, len(wcss) - 1):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        dis.append(numerator / denominator)
    
    elbow_point = dis.index(max(dis)) + 2

    return elbow_point, wcss

def elbow_plot(elbow_point, wcss, file_name, algorithm, threshold):
    plt.clf()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.axvline(elbow_point, color='b', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title(f'{file_name}_{threshold}_Elbow Method')

# Algorithm for choosing the number of clusters. 
def choose_cluster(elbow, silhouette):
    chosen_cluster = 0
    cluster_info = ""

    # If the values of elbow method and silhouette method are different, 
    if elbow != silhouette:
        cluster_info += f"Elbow method suggests {elbow} clusters.\n"
        cluster_info += f"Silhouette method suggests {silhouette} clusters.\n"

        if abs(elbow - silhouette) > 1:
            cluster_info += "The difference between the two methods is significant.\n"
            cluster_info += " Choosing the number of clusters based on silhouette method.\n"
            chosen_cluster = silhouette
        
        else:
            cluster_info += "The difference between the two methods is not significant."
            cluster_info += " Choosing the number of clusters based on their average."
            chosen_cluster = int((elbow + silhouette) / 2)
    
    else:
        cluster_info += f"Both methods suggest the same number of clusters: {elbow}.\n"
        chosen_cluster = elbow
    
    return chosen_cluster, cluster_info

# Perform k-Means clustering algorithm
def kmeans(data, n_cluster):
    kmean = KMeans(n_clusters = n_cluster, random_state=42, n_init='auto')
    labels = kmean.fit_predict(data)

    return labels

# Perform Hierarchical clustering, Agglomerative (aka bottom-up method) algorithm
def agglomerative(data, n_cluster):
    agg_clustering = AgglomerativeClustering(n_clusters = n_cluster).fit(data)
    labels = agg_clustering.labels_

    return labels

# Choose which clustering algorithm will be run, depend on the user's choice
def choose_algo(data, n_cluster, algorithm):
    if algorithm == 'k-Means':
        return kmeans(data, n_cluster)
    
    elif algorithm == 'Agglomerative':
        return agglomerative(data, n_cluster)
    
    else:
        return kmeans(data, n_cluster), agglomerative(data, n_cluster)

# Generate the cluster plots, depending on the user's choice
def plot_cluster(pca_df, file_name, algorithm, threshold):
    
    # If user choose only Agglomerative clustering algorithm, then plot agglomerative cluster
    if 'Agglomerative Cluster' in pca_df.columns:
        axs = plt.subplots()
        axs = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='Agglomerative Cluster', data=pca_df)
        plt.title(f'{file_name} {threshold} {algorithm} Cluster')
        #plt.savefig(f'./static/_img/{file_name}_{threshold}_Agglomerative_Cluster.png')
    
    # If user choose only k-Means clustering algorithm, then plot k-Means cluster
    if 'k-Means Cluster' in pca_df.columns:
        axs = plt.subplots()
        axs = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='k-Means Cluster', data=pca_df)
        plt.title(f'{file_name} {threshold} {algorithm} Cluster')
        #plt.savefig(f'./static/_img/{file_name}_{threshold}_k-Means_Cluster.png')

# Determine optimal number of clusters using silhouette method
class silhouetteAnalyze:
    def __init__(self, data):
        self.data = data
        self.silhouette_scores = None
        self.optimal_clusters = None
    
    def analyze(self):
        silhouette_scores = []
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        self.silhouette_scores = silhouette_scores
    
    def get_optimal_clusters(self):
        if self.silhouette_scores is None:
            print("Call analyze() method first to compute silhouette scores.")
            return None

        optimal_clusters_index = np.argmax(self.silhouette_scores)
        self.optimal_clusters = optimal_clusters_index + 2
        return self.optimal_clusters
    
    def get_silhouette_scores(self):
        if self.silhouette_scores is None:
            print("Call analyze() method first to compute silhouette scores.")
            return None
        
        return self.silhouette_scores
    
    def plot(self, file_name, algorithm, threshold):
        if self.silhouette_scores is None:
            print("Call analyze() method first to compute silhouette scores.")
            return
        
        plt.clf()
        plt.plot(range(2, 11), self.silhouette_scores, marker='o')
        plt.axvline(self.optimal_clusters, color='b', linestyle='-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'{file_name}_{threshold}_Silhouette Method')
