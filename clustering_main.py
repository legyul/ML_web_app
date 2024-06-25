from clustering import load_file, Path, preprocessing_data, identify_variable, elbow, elbow_plot, silhouetteAnalyze, choose_cluster, choose_algo, perform_pca, plot_cluster, pd, plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


def main(file_key, threshold, algorithm, plot):

    # Call the file and save it to a variable, df
    df = load_file(file_key)
    file_name = Path(file_key).stem
    print("Algorithm : ", algorithm)
    # Create a PDF document
    doc = SimpleDocTemplate(f"{file_name}_{algorithm}_{threshold}_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12))

    title = Paragraph("Clustering Report", styles['Title'])
    file_name_para = Paragraph(f"File Name: {file_name}", styles['Normal'])

    pre_df, label, unique = preprocessing_data(df)

    if len(label) != 0:
        print('label:', label)
    
    if len(unique) != 0:
        print('unique: ', unique)

    # Reduce columns with the most relevant columns
    variable_names = identify_variable(pre_df, threshold)
    filtered_df = pre_df[variable_names]
    useful_variable = f"Use {variable_names} to cluster."

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
        
        if 'Agglomerative Cluster' in df.columns:
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

    variable_text = Paragraph(useful_variable, styles['Normal'])

    cluster_text = Paragraph(cluster_info, styles['Normal'])
    silhouette_text = Paragraph(silhouette_info, styles['Normal'])
    score_text = Paragraph(score_info, styles['Normal'])
    line = HRFlowable(width="100%", thickness=1, lineCap='square', color="black", spaceBefore=10, spaceAfter=10)

    content = [title, Spacer(1, 24), file_name_para, line, variable_text, Spacer(1, 12), Spacer(1, 12), cluster_text,
                silhouette_text, score_text, Spacer(1,12)]

    if len(label) != 0 and len(unique) != 0:
        combined_text = ", ".join(f"{l}: {u}" for l, u in zip(label, unique))
        combined_paragraph = Paragraph(combined_text, styles['Normal'])
        content.insert(6, combined_paragraph)

    # Print 5 samples of each cluster.
    # kmeans_clusters = df['k-Means Cluster'].unique()
    # agglom_clusters = df['Agglomerative Cluster'].unique()
    # k_df = df.drop(columns=['Agglomerative Cluster'])
    # a_df = df.drop(columns=['k-Means Cluster'])

    # page_width, _ = landscape(letter)
    # col_width = page_width / len(k_df.columns)

    # if algorithm == 'both':
    #     kmeans_text = Paragraph("k-Means Cluster samples:\n", styles['Bold'])
    #     content.append(kmeans_text)
    #     for cluster in kmeans_clusters:
    #         k_samples = k_df[k_df['k-Means Cluster'] == cluster].sample(n=5, replace=True)
            
    #         # Add cluster header in bold
    #         cluster_header = Paragraph(f"Cluster {cluster} samples:", styles['Normal'])
    #         content.append(cluster_header)
            
    #         # Convert samples to table
    #         data = [k_samples.columns.values.tolist()] + k_samples.values.tolist()
    #         table = Table(data, colWidths=[min(col_width, 50)]*len(k_samples.columns))
    #         table.setStyle(TableStyle([
    #             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    #             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    #             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #             ('FONTSIZE', (0, 0), (-1, -1), 6),  # Adjust font size to fit content
    #             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    #             ('GRID', (0, 0), (-1, -1), 1, colors.black),
    #             ('WORDWRAP', (0, 0), (-1, -1), 'CJK')
    #         ]))
    #         content.append(table)
    #         content.append(Spacer(1, 12))
        
    #     agglom_text = Paragraph("Agglomerative Cluster samples:\n", styles['Bold'])
    #     content.append(agglom_text)
    #     for cluster in agglom_clusters:
    #         a_samples = a_df[a_df['Agglomerative Cluster'] == cluster].sample(n=5, replace=True)
            
    #         # Add cluster header in bold
    #         cluster_header = Paragraph(f"Cluster {cluster} samples:", styles['Normal'])
    #         content.append(cluster_header)
            
    #         # Convert samples to table
    #         data = [a_samples.columns.values.tolist()] + a_samples.values.tolist()
    #         table = Table(data, colWidths=[min(col_width, 50)]*len(a_samples.columns))
    #         table.setStyle(TableStyle([
    #             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    #             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    #             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #             ('FONTSIZE', (0, 0), (-1, -1), 6),  # Adjust font size to fit content
    #             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    #             ('GRID', (0, 0), (-1, -1), 1, colors.black),
    #             ('WORDWRAP', (0, 0), (-1, -1), 'CJK')
    #         ]))
    #         content.append(table)
    #         content.append(Spacer(1, 12))

    # elif algorithm == 'k-Means':
    #     kmeans_text = Paragraph("k-Means Cluster samples:\n", styles['Normal'])
    #     content.append(kmeans_text)
    #     for cluster in kmeans_clusters:
    #         k_samples = k_df[k_df['k-Means Cluster'] == cluster].sample(n=5, replace=True)
            
    #         # Add cluster header in bold
    #         cluster_header = Paragraph(f"Cluster {cluster} samples:", styles['Bold'])
    #         content.append(cluster_header)
            
    #         # Convert samples to table
    #         data = [k_samples.columns.values.tolist()] + k_samples.values.tolist()
    #         table = Table(data, colWidths=[min(col_width, 50)]*len(k_samples.columns))
    #         table.setStyle(TableStyle([
    #             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    #             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    #             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #             ('FONTSIZE', (0, 0), (-1, -1), 6),  # Adjust font size to fit content
    #             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    #             ('GRID', (0, 0), (-1, -1), 1, colors.black),
    #             ('WORDWRAP', (0, 0), (-1, -1), 'CJK')
    #         ]))
    #         content.append(table)
    #         content.append(Spacer(1, 12))

    # elif algorithm == 'Agglomerative':
    #     agglom_text = Paragraph("Agglomerative Cluster samples:\n", styles['Normal'])
    #     content.append(agglom_text)
    #     for cluster in agglom_clusters:
    #         a_samples = a_df[a_df['Agglomerative Cluster'] == cluster].sample(n=5, replace=True)
            
    #         # Add cluster header in bold
    #         cluster_header = Paragraph(f"Cluster {cluster} samples:", styles['Bold'])
    #         content.append(cluster_header)
            
    #         # Convert samples to table
    #         data = [a_samples.columns.values.tolist()] + a_samples.values.tolist()
    #         table = Table(data, colWidths=[min(col_width, 50)]*len(a_samples.columns))
    #         table.setStyle(TableStyle([
    #             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    #             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    #             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    #             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #             ('FONTSIZE', (0, 0), (-1, -1), 6),  # Adjust font size to fit content
    #             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    #             ('GRID', (0, 0), (-1, -1), 1, colors.black),
    #             ('WORDWRAP', (0, 0), (-1, -1), 'CJK')
    #         ]))
    #         content.append(table)
    #         content.append(Spacer(1, 12))

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