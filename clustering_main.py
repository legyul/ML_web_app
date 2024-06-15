from clustering import load_file, Path, preprocessing_data, identify_variable, elbow, elbow_plot, silhouetteAnalyze, choose_cluster, choose_algo, perform_pca, plot_cluster, pd, plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


def main(file_key, threshold, algorithm, plot):

    # Call the file and save it to a variable, df
    df = load_file(file_key)
    file_name = Path(file_key).stem
    print("Algorithm : ", algorithm)
    # Create a PDF document
    doc = SimpleDocTemplate(f"{file_name}_{algorithm}_{threshold}_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    title = Paragraph("Clustering Report", styles['Title'])
    file_name_para = Paragraph(f"File Name: {file_name}", styles['Normal'])

    pre_df = preprocessing_data(df)

    # Reduce columns with the most relevant columns
    variable_names = identify_variable(pre_df, threshold)
    filtered_df = pre_df[variable_names]

    # Elbow method to determine the number of clusters
    elbow_cluster, wcss = elbow(filtered_df)
    elbow_plot(elbow_cluster, wcss, file_name, algorithm, threshold)

    # Silhouette method to determine the number of clusters
    silhouette = silhouetteAnalyze(filtered_df)
    silhouette.analyze()
    silhou_cluster = silhouette.get_optimal_clusters()
    silhouette.plot(file_name, algorithm, threshold)

    n_cluster, cluster_info = choose_cluster(elbow_cluster, silhou_cluster)

    cluster = choose_algo(filtered_df, n_cluster, algorithm)
    if algorithm == 'both':
        kmeans_label, agglom_label = cluster
        df['k-Means Cluster'] = kmeans_label
        df['Agglomerative Cluster'] = agglom_label

    elif algorithm == 'k-Means':
        kmeans_label = cluster
        df['k-Means Cluster'] = kmeans_label
        
    elif algorithm == 'Agglomerative':
        agglom_label = cluster
        df['Agglomerative Cluster'] = agglom_label

    if plot == 'yes':
        pca = perform_pca(filtered_df)
        
        pca_df = pd.DataFrame(pca)
        if 'k-Means Cluster' in df.columns:
            pca_df['k-Means Cluster'] = df['k-Means Cluster']
        
        elif 'Agglomerative Cluster' in df.columns:
            pca_df['Agglomerative Cluster'] = df['Agglomerative Cluster']

        plot_cluster(pca_df, file_name, algorithm, threshold)

        scores = silhouette.get_silhouette_scores()
        scores = scores[n_cluster - 2]
        
        if scores >= 0.5:
            score_info = "Since Silhouette Score is greater than or equal to 0.5, it is STRONG evidence of well-defined clusters."
        
        elif scores < 0.5 and scores >= 0.25:
            score_info = "Since Silhouette Score is in between 0.25 and 0.5, it is MODERATE evidence of well-defined clusters."
        
        elif scores < 0.25 and scores > -0.25:
            score_info = "Since Silhouette Score is in between -0.25 and 0.25, it is WEAK or No structure."

        elif scores <= -0.25 and scores > -0.5:
            score_info = "Since Silhouette Score is in between -0.5 and -0.25, it May indicate that data points have been assigned to the wrong clusters."

        elif scores <= -0.5:
            score_info = "Since Silhouette Score is smaller than or eqaul to -0.5, it is Strong evidence of misclassification or poor clustering structure."
    
    silhouette_info = f"\nSilhouette Score: {scores}"

    cluster_text = Paragraph(cluster_info, styles['Normal'])
    silhouette_text = Paragraph(silhouette_info, styles['Normal'])
    score_text = Paragraph(score_info, styles['Normal'])
    line = HRFlowable(width="100%", thickness=1, lineCap='square', color="black", spaceBefore=10, spaceAfter=10)

    content = [title, Spacer(1, 24), file_name_para, line, cluster_text, silhouette_text, score_text, Spacer(1,12)]

    elbow_img = f'./static/_img/{file_name}_{threshold}_{algorithm}_Elbow_Method.png'
    if elbow_img:
        plot_img = plt.imread(elbow_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        elbow_img_obj = Image(elbow_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(elbow_img_obj)

    silhouette_img = f'./static/_img/{file_name}_{threshold}_{algorithm}_Silhouette_Method.png'
    if silhouette_img:
        plot_img = plt.imread(silhouette_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        silhouette_img_obj = Image(silhouette_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(silhouette_img_obj)

    kmeans_img = f'./static/_img/{file_name}_{threshold}_k-Means_Cluster.png'
    agglom_img = f'./static/_img/{file_name}_{threshold}_Agglomerative_Cluster.png'

    if algorithm == 'both':
        # k-Means plot
        plot_img = plt.imread(kmeans_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        kmeans_img_obj = Image(kmeans_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(kmeans_img_obj)

        # Agglomerative plot
        plot_img = plt.imread(agglom_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        agglom_img_obj = Image(agglom_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(agglom_img_obj)

    elif algorithm == 'Agglomerative':
        plot_img = plt.imread(agglom_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        agglom_img_obj = Image(agglom_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(agglom_img_obj)

    elif algorithm == 'k-Means':
        plot_img = plt.imread(kmeans_img)
        img_width = 400
        img_height = img_width * plot_img.shape[0] / plot_img.shape[1]
        kmeans_img_obj = Image(kmeans_img, width=img_width, height=img_height)
        content.append(Spacer(1, 12))
        content.append(kmeans_img_obj)
    
    doc.build(content)
    df.to_csv(f'{file_name}_{algorithm}_{threshold}_Result.csv')

    csv_path = f'{file_name}_{algorithm}_{threshold}_Result.csv'
    pdf_path = f'{file_name}_{algorithm}_{threshold}_Report.pdf'

    return pdf_path, csv_path

# main('./data/wine-clustering.csv', 0.5, 'both', 'yes')