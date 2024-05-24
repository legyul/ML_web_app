# Load libraries
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from dotenv import load_dotenv
import os
import boto3

load_dotenv()

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Functions
# Load dataset file
def load_file(file_key):
    file_path = f"/tmp/{file_key}"
    s3.download_file(S3_BUCKET_NAME, file_key, file_path)
    file_extention = file_path.split('.')[-1]
    if file_extention == 'csv':
        return pd.read_csv(file_path)
    elif file_extention == 'xlsx':
        return pd.read_excel(file_path)
    elif file_extention == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format, use .csv, .xlsx, or .json")

# Find the useful variables to cluster
def identify_variable(data, threshold):
    corr_matrix = data.corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    variables = (upper_triangle[upper_triangle > threshold]).stack().index.tolist()
    variables = list(set([item for sublist in variables for item in sublist]))

    return variables

def preprocessing_data(data):
    data.fillna(data.mean(), inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    return scaled_df

def perform_pca(data):
    pca = PCA(n_components=2)
    transform_data = pca.fit_transform(data)

    return transform_data

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
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.axvline(elbow_point, color='b', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.savefig('./static/_img/Elbow_Method.png')
    plt.show()

    return elbow_point

# Algorithm for choosing the number of clusters. 
def choose_cluster(elbow, silhouette):
    cluster_info = ""

    # If the values of elbow method and silhouette method are different, 
    if elbow != silhouette:
        cluster_info += f"Elbow method suggests {elbow} clusters.\n"
        cluster_info += f"Silhouette method suggests {silhouette} clusters.\n"

        if abs(elbow - silhouette) > 1:
            cluster_info += "The difference between the two methods is significant.\n"
            cluster_info += "Choosing the number of clusters based on silhouette method.\n"
            chosen_cluster = silhouette
        
        else:
            cluster_info += "The difference between the two methods is not significant."
            cluster_info += "Choosing the number of clusters based on their average."
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
def choose_algo(data, n_cluster, algorithm='both'):
    if algorithm == 'k-Means':
        return kmeans(data, n_cluster)
    
    elif algorithm == 'Agglomerative Clustering':
        return agglomerative(data, n_cluster)
    
    else:
        return kmeans(data, n_cluster), agglomerative(data, n_cluster)

# Generate the cluster plots, depending on the user's choice
def plot_cluster(pca_df):
    # If user choose both algorithms, plot both
    if 'k-Means Cluster' and 'Agglomerative Cluster' in pca_df.columns:
        axs_k = plt.subplots()
        axs_k = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='k-Means Cluster', data=pca_df)
        plt.title('k-Means Cluster')
        plt.savefig('./static/_img/k-Means_Cluster.png')

        axs_a = plt.subplots()
        axs_a = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='Agglomerative Cluster', data=pca_df)
        plt.title('Agglomerative Cluster')
        plt.savefig('./static/_img/Agglomerative_Cluster.png')
    
    # If user choose only Agglomerative clustering algorithm, then plot agglomerative cluster
    elif 'Agglomerative Cluster' in pca_df.columns:
        axs = plt.subplots()
        axs = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='Agglomerative Cluster', data=pca_df)
        plt.title('Agglomerative Cluster')
        plt.savefig('./static/_img/Agglomerative_Cluster.png')
    
    # If user choose only k-Means clustering algorithm, then plot k-Means cluster
    elif 'k-Means Cluster' in pca_df.columns:
        axs = plt.subplots()
        axs = sns.scatterplot(x=pca_df[0], y=pca_df[1], hue='k-Means Cluster', data=pca_df)
        plt.title('k-Means Cluster')
        plt.savefig('./static/_img/k-Means_Cluster.png')

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
    
    def plot(self):
        if self.silhouette_scores is None:
            print("Call analyze() method first to compute silhouette scores.")
            return
        
        plt.plot(range(2, 11), self.silhouette_scores, marker='o')
        plt.axvline(self.optimal_clusters, color='b', linestyle='-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        plt.savefig('./static/_img/Silhouette_Method.png')
        plt.show()

