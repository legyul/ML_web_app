from models import common
from .clustering import filter_data, elbow, elbow_plot, silhouetteAnalyze, choose_cluster, choose_algo, visualize_pca, plot_cluster, pd, plt
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
import io
from reportlab.lib.utils import ImageReader


def run_cluster(file_key, threshold, algorithm, plot):

    # Call the file and save it to a variable, df
    df, mode = common.load_file(file_key)
    if mode == 'spark':
        df = df.na.drop(how='all')
    else:
        df = df.dropna(how='all')
    file_name = Path(file_key).stem
    
    # Create a PDF document
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12))

    title = Paragraph("Clustering Report", styles['Title'])
    file_name_para = Paragraph(f"File Name: {file_name}", styles['Normal'])

    pre_df, gender_mapping = common.spark_processing.spark_preprocessing_data(df, mode)

    unique = list(gender_mapping.keys())
    label = list(gender_mapping.values())

    # Reduce columns with the most relevant columns
    filtered_df, variables, pca_info = filter_data(pre_df, threshold_corr=0.85, threshold_var=0.02, explained_variance=0.95, max_components=10)

    useful_variable = f"Use {variables.columns} to cluster. {pca_info}"

    # Elbow method to determine the number of clusters
    elbow_cluster, wcss = elbow(filtered_df)


    # Silhouette method to determine the number of clusters
    silhouette = silhouetteAnalyze(filtered_df)
    silhouette.analyze()
    silhou_cluster = silhouette.get_optimal_clusters()

    # silhouette.plot(file_name, algorithm, threshold)

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

    image_buffers = []

    def add_plot_to_pdf(plot_func, *args):
        img_buffer = io.BytesIO()
        plot_func(*args)
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_buffer.seek(0)
        image_buffers.append(img_buffer)
    add_plot_to_pdf(elbow_plot, elbow_cluster, wcss, file_name, algorithm, threshold)
    add_plot_to_pdf(silhouette.plot, file_name, algorithm, threshold)

    if plot == 'yes':
        pca = visualize_pca(filtered_df, mode)
        
        pca_df = pd.DataFrame(pca)
        if 'k-Means Cluster' in df.columns:
            pca_df['k-Means Cluster'] = df['k-Means Cluster']
        
        if 'Agglomerative Cluster' in df.columns:
            pca_df['Agglomerative Cluster'] = df['Agglomerative Cluster']
        
        if algorithm == 'both':
            add_plot_to_pdf(plot_cluster, pca_df, file_name, "k-Means", threshold)
            add_plot_to_pdf(plot_cluster, pca_df, file_name, "Agglomerative", threshold)
        elif algorithm =='k-Means':
            add_plot_to_pdf(plot_cluster, pca_df, file_name, "k-Means", threshold)
        elif algorithm == 'Agglomerative':
            add_plot_to_pdf(plot_cluster, pca_df, file_name, "Agglomerative", threshold)


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

    pca_text = Paragraph(pca_info, styles['Normal'])
    variable_text = Paragraph(useful_variable, styles['Normal'])

    cluster_text = Paragraph(cluster_info, styles['Normal'])
    silhouette_text = Paragraph(silhouette_info, styles['Normal'])
    score_text = Paragraph(score_info, styles['Normal'])
    line = HRFlowable(width="100%", thickness=1, lineCap='square', color="black", spaceBefore=10, spaceAfter=10)

    content = [title, Spacer(1, 24), file_name_para, line, pca_text, variable_text, Spacer(1, 12), Spacer(1, 12), cluster_text,
                silhouette_text, score_text, Spacer(1,12)]

    if len(label) != 0 and len(unique) != 0:
        combined_text = ", ".join(f"{l}: {u}" for l, u in zip(label, unique))
        combined_paragraph = Paragraph(combined_text, styles['Normal'])
        content.insert(6, combined_paragraph)
    
    for img_buffer in image_buffers:
        content.append(Spacer(1, 12))
        content.append(Image(img_buffer, width=400, height=300))


    doc.build(content)

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return pdf_buffer, csv_buffer
