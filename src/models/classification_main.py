from .common import load_file
from .classification_models import preprocess, select_model, build_model_dict, BestModel, individual_model
from model_utils import save_model_with_info
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
import io
import zipfile

def run_classification(file_key, model_choice):
    file_name = Path(file_key).stem

    model_info_buffer = io.BytesIO()
    model_buffer = io.BytesIO()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Bold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12))

    title = Paragraph(f"Classification {model_choice} Report", styles['Title'])
    file_name_para = Paragraph(f"File Name: {file_name}", styles['Normal'])

    df, _ = load_file(file_key)

    df, target, language_column, bow_list = preprocess.preprocess_text_columns(df)

    regression = preprocess.is_continuous_data  # T- regression, F - classification
    
    mode = None
    if regression == True:
        mode = 'regression'
    else:
        mode = 'classification'
    
    X = df.drop(columns=target)
    y = df[target]

    required_packages = [
        'numpy',
        'pandas'
    ]

    if model_choice in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'Logistic Regression']:
        best_model, model_accuracy, model_score, y_type, label_map = individual_model(model_choice, X, y, mode=mode)
        model_name = model_choice
        model_scores = f"Accuracy: {model_accuracy: .4f}, Score: {model_score: .4f}"
    
    elif model_choice == 'Find Best Model':
        models, LR_best_params, LR_tuned_scores = build_model_dict(X, y)
        best_model_name, model_names, best_model, best_score, label_map, y_type = select_model.model_selection(models, X, y, mode=mode, k=5)
        model_name = best_model_name
        
        model_scores = f"Score with {model_name}: {best_score: .4f}"
    
    if y_type == 'categorical':
        model = BestModel(model=best_model, label_mapping=label_map)
        
        model_info_buffer, model_buffer = save_model_with_info(model=model, model_name=model_name, required_packages=required_packages)
    
    else:
        model_info_buffer, model_buffer = save_model_with_info(model=best_model, model_name=model_name, required_packages=required_packages)
    
    if regression:
        continuous_data = 'The Dataset is continuous. Use Regression'
    else:
        continuous_data = 'The Dataset is not continuouse. Use Classification'
    
    if language_column is None:
        language_column = []

    if len(language_column) > 0:
        language_column_info = f"Text columns in your dataset: {language_column}"
        bow_list_info = f"Your dataset words list: {bow_list}"
    else:
        language_column_info = f"There are no text columns in your dataset"
        bow_list_info = ""
    
    continuous_text = Paragraph(continuous_data, styles['Normal'])
    language_column_text = Paragraph(language_column_info, styles['Normal'])
    bow_list_text = Paragraph(bow_list_info, styles['Normal'])
    model_scores_text = Paragraph(model_scores, styles['Normal'])
    info = 'For more detailed information, please check the log file.'
    info_text = Paragraph(info, styles['Normal'])

    content = []
    line = HRFlowable(width="100%", thickness=1, lineCap='square', color="black", spaceBefore=10, spaceAfter=10)
    
    if model_choice == 'Find Best Model':
        used_model_names = f"{model_name} with the highest score among {model_names}"
        used_model_names_text = Paragraph(used_model_names, styles['Normal'])

        content = [title, Spacer(1, 24), file_name_para, line, continuous_text, language_column_text, bow_list_text,
                    Spacer(1, 12), model_scores_text, used_model_names_text, Spacer(1, 12), Spacer(1, 12), info_text]
    
    else:
        content = [title, Spacer(1, 24), file_name_para, line, continuous_text,language_column_text, bow_list_text,
                   Spacer(1, 12), model_scores_text, Spacer(1, 12), Spacer(1, 12), info_text]

    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        doc.build(content)

        pdf_buffer.seek(0)
        model_info_buffer.seek(0)
        model_buffer.seek(0)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model info and model file to the zip buffer
            zipf.writestr(f"{file_name}_model_info.json", model_info_buffer.getvalue())  # Model info file
            zipf.writestr(f"{file_name}_model.pkl", model_buffer.getvalue())  # Model file

        
        zip_buffer.seek(0)
        print('\n\nDone!!!!!\n\n')

        return pdf_buffer, zip_buffer
    except Exception as e:
        print(f"Error during {file_name}: {e}")
        return None, None
