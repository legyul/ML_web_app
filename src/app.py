import os
import sys
from models import run_cluster, run_classification #, common
from logger_utils import setup_global_logger
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify, session, Response
from werkzeug.utils import secure_filename
import boto3
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint
import logging
from pyspark import SparkConf, SparkContext
import io
import pandas as pd
from fpdf import FPDF
import zipfile
import pickle

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

template_dir = os.path.abspath('./templates')
static_dir = os.path.abspath('./static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise ValueError("FLASK_SECRET_KEY is not set! Set the environment variable before running the app.")

# Setting the port that Spark UI uses
conf = SparkConf().setAppName("DataPreprocessing").setMaster("local[*]")

if SparkContext._active_spark_context:
    SparkContext._active_spark_context.stop()

sc = SparkContext.getOrCreate(conf=conf)

S3_REGION = "us-east-2"
# S3 Client configuration
s3 = boto3.client('s3', region_name=S3_REGION, config=boto3.session.Config(signature_version='s3v4'))

S3_BUCKET_NAME = "ml-platform-service"

UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "ML Platform"
    }
)

logger, upload_log_to_s3 = setup_global_logger(s3, bucket_name=S3_BUCKET_NAME)#, log_filename=current_filename)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

current_filename = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clustering')
def clustering():
    return render_template('upload_clustering.html')

@app.route('/classification')
def classification():
    return render_template('upload_classification.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global logger, upload_log_to_s3
    print("[DEBUG] Implementing upload_file()")
    if 'file' not in request.files:
        print("[ERROR] No file")
        return redirect(request.url)
    
    file = request.files['file']
    task = request.form.get('task')
    
    if file.filename == '':
        print("[ERROR] No file name")
        return redirect(request.url)
    
    if file:
        global filename
        filename = file.filename
        print(f"[DEBUG] File name to upload: {filename}")

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger, upload_log_to_s3 = setup_global_logger(s3, bucket_name=S3_BUCKET_NAME, log_filename=filename)

        # Use the existing function to upload the file directly to S3
        s3_file_path = f"uploaded/{file.filename}"

        # Call the upload function with the file, S3 bucket name, and the target file
        upload_user_file_to_s3(file, S3_BUCKET_NAME, file.filename)

        print(f"File {file.filename} uploaded to S3 bucket {S3_BUCKET_NAME}")
        
        # File uploaded, now decide what to do based on the selected task
        if task == 'clustering':
            print("clustering chose")
            return redirect(url_for('process_clustering', filename=file.filename))

        
        # User choose classification option
        elif task == 'classification':
           return redirect(url_for('process_classification', filename=file.filename))
    
    return redirect(url_for('index'))


def delete_file_from_s3(bucket_name, file_key):
    s3.delete_object(Bucket=bucket_name, Key=file_key)
    print(f"File {file_key} deleted from S3 bucket {bucket_name}")

@app.route('/process_clustering/<filename>', methods=['GET', 'POST'])
def process_clustering(filename):
    try:
        s3_file_path = f"s3://{S3_BUCKET_NAME}/uploaded/{filename}"
        
        if request.method == 'POST':
            threshold = float(request.form.get('threshold'))
            algorithm = request.form.get('algorithm')
            plot = request.form.get('plot')

            if not threshold or not algorithm or not plot:
                raise ValueError("Missing required parameters.")
            
            threshold = float(threshold)

            # Implement main function and generate report and result file
            pdf_file, csv_file = run_cluster(s3_file_path, threshold, algorithm, plot)

            files_to_upload = {
                f"{filename}_report.pdf": pdf_file,
                f"{filename}_results.csv": csv_file
            }

            # Upload the generated report and result file directly to S3
            print("\nUploading files to S3...\n")
            upload_to_s3_direct(S3_BUCKET_NAME, files_to_upload)
            print("\nUplad function executed.\n")

            # Generate presigned URL
            pdf_url = generate_presigned_url(S3_BUCKET_NAME, f"result/{filename}_report.pdf")
            csv_url = generate_presigned_url(S3_BUCKET_NAME, f"result/{filename}_results.csv")

            print(f"PDF URL: {pdf_url}")
            print(f"CSV URL: {csv_url}")

            return render_template('clustering_result.html', pdf_url=pdf_url, csv_url=csv_url)
        
    # If file extention is not suported, delet the file from S3 Bucket
    except ValueError as e:
        flash(f"Invalid input: {str(e)}", "error")

        delete_file_from_s3(S3_BUCKET_NAME, f"uploaded/{filename}")
        return redirect(request.url)
    
    except FileNotFoundError as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(request.url)
    
    except Exception as e:
        flash("An unexpected error occurred. Pleas try again later.", "error")
        print(f"Unexpected Error: {str(e)}")
        return redirect(request.url)

    return render_template('process_clustering.html', filename=filename)

@app.route('/process_classification/<filename>', methods=['GET', 'POST'])
def process_classification(filename):
    if request.method == 'POST':
        model_choice = request.form.get('model')

        if not model_choice:
            flash("Please select a model.")
            return redirect(request.url)
        
        # After choose the model, move to loading page
        return render_template('loading.html', filename=filename, model_choice=model_choice)
    return render_template('select_model.html', filename=filename)

@app.route('/start_classification/<filename>', methods=['POST'])
def start_classification(filename):
    global progress_status
    data = request.json
    #filename = data.get('filename')
    model_choice = data.get("model_choice")

    if not filename or not model_choice:
        return jsonify({"error": "Missing filename or model choice"}), 400

    
    s3_file_path = f"uploaded/{filename}"
    progress_status = "Training started..."

    try:
        try:
            pdf_file, model_buffer = run_classification(s3_file_path, model_choice=model_choice)
        except Exception as e:
            return jsonify({"error": "Error during classification processing."}), 500

        model_filename = f'{filename}_{model_choice}_model_and_info.zip'
        pdf_filename = f'{filename}_{model_choice}_Report.pdf'

        files_to_uploads = {
            model_filename: model_buffer,
            pdf_filename: pdf_file
        }

        upload_to_s3_direct(S3_BUCKET_NAME, files=files_to_uploads)
        upload_log_to_s3()

        # Generate Download URL
        model_url = generate_presigned_url(S3_BUCKET_NAME, f"result/{filename}_{model_choice}_model_and_info.zip")
        pdf_url = generate_presigned_url(S3_BUCKET_NAME, f"result/{filename}_{model_choice}_Report.pdf")
        log_url = generate_presigned_url(S3_BUCKET_NAME, f"logs/{filename}_log.log")

        session['pdf_url'] = pdf_url
        session['model_url'] = model_url
        session['log_url'] = log_url

        print("\n=== Session Data Saved ===")
        print("PDF URL:", session.get('pdf_url'))
        print("Model URL:", session.get('model_url'))
        print("Log URL:", session.get('log_url'))

        progress_status = "Classification completed!"
        return jsonify({"message": "Classification completed. You can now view the results."})
    
    except Exception as e:
        progress_status = "Error occurred!"
        print(f"\n=== Error in start_classification ===\n{e}\n")
        return jsonify({"Error": str(e)}), 500

progress_status = "Waiting..."

@app.route('/classification_result')
def classification_result():
    pdf_url = session.get('pdf_url')
    model_url = session.get('model_url')
    log_url = session.get('log_url')

    print("\n=== Accessing classification_result ===")
    print("PDF URL:", pdf_url)
    print("Model URL:", model_url)
    print("Log URL:", log_url)

    if not pdf_url or not model_url or not log_url:
        flash("Error: Missing classification result data. Please try again.")
        return redirect(url_for('home'))
    
    return render_template(
        'classification_result.html',
        pdf_url=pdf_url,
        model_url=model_url,
        log_url=log_url
    )

@app.route('/progress')
def progress():
    global progress_status
    return jsonify({"status": progress_status})

@app.route('/view_log/<filename>')
def view_log():
    log_s3_key = f"logs/{filename}"

    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=log_s3_key)
        log_content = response['Body'].read().decode('utf-8')

        print("\n[DEBUG] Successfully retrieved log content!")
        return render_template('view_log.html', log_content=log_content.splitlines(), filename=filename)
    
    except s3.exceptions.NoSuchKey:
        print(f"[ERROR] Log file '{log_s3_key}' not found in S3.")  # 디버깅 로그
        return "Log file not found.", 404

    except Exception as e:
        print(f"[ERROR] Error retrieving log file: {str(e)}")  # 디버깅 로그
        return f"Error retrieving log file: {str(e)}", 500

@app.route('/download_log/<filename>')
def download_log(filename):
    log_s3_key = f"logs/{filename}"

    try:
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=log_s3_key)
        log_content = response['Body'].read()
        
        # 파일 다운로드 응답 설정
        return Response(
            log_content,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )
    except s3.exceptions.NoSuchKey:
        return "Log file not found.", 404
    except Exception as e:
        return f"Error retrieving log file: {str(e)}", 500

# Upload generated files to S3 bucket
def upload_to_s3_direct(bucket_name, files):
    '''
    Upload the file data directly to S3 without saving it locally

    Parameters
    - file_name: The name of the file to be uploaded
    - bucket_name: The name of the S3 bucket
    - files (dict): The content to upload, in byt-like format (e.g., byte string, file object)
    '''
    # Iterate through the files and upload each one to S3
    for file_name, file_data in files.items():
        file_buffer = io.BytesIO()

        # Handle different file types
        if isinstance(file_data, io.BytesIO):
            file_buffer = file_data
            print(f"\nCOnvertin {file_name} to ...")
        elif isinstance(file_data, pd.DataFrame):     # For CSV
            print(f"\nCOnverting {file_name} to CSV format...")
            file_data.to_csv(file_buffer, index=False)
        elif isinstance(file_data, type(FPDF())):   # For PDF
            print(f"\nGenerating PDF: {file_name}...\n")
            file_data.output(file_buffer)
        elif isinstance(file_data, str):    # For log or other text files
            print(f"\nProcessing log file: {file_name}")
            file_buffer.write(file_data.encode('utf-8'))
        elif isinstance(file_data, bytes):      # If file is already in bytes (like model.pkl)
            print(f"\nProcessing model file: {file_name}...\n")
            file_buffer.write(file_data)
        else:
            pickle.dump(file_data, file_buffer)
        
        file_buffer.seek(0)

        try:
            # Upload the file to S3
            print(f"Uploading {file_name} to S3...")
            s3.upload_fileobj(file_buffer, bucket_name, f'result/{file_name}')
            print(f"File {file_name} uploaded to S3 bucket {bucket_name}.")
        
        except Exception as e:
            print(f"Error uploading {file_name} to S3: {e}")

def upload_user_file_to_s3(file, bucket_name, file_name):
    '''
    Upload user-provided file to S3

    Parameters
    - file: The file object uploaded by the user (request.files['file'])
    - bucket_name: The name of the S3 bucket
    - file_name: The name to store the file as in the S3 bucket
    '''
    try:
        # Convert the file to BytesIO (in-memory file-like object)
        file_buffer = io.BytesIO()
        file.save(file_buffer)
        file_buffer.seek(0)
        print(f"[DEBUG] File {file_name} loaded into memory")

        file_size = len(file_buffer.getvalue())
        print(f"[DEBUG] File size: {file_size} bytes")
        print(f"[DEBUG] Uploading {file_name} to S3 at path: uploaded/{file_name}")
        # Upload the file to S3
        s3.upload_fileobj(file_buffer, bucket_name, f'uploaded/{file_name}')#, ExtraArgs={'ACL':'public-read'})
        print(f"File {file_name} uploaded to S3 bucket {bucket_name}.")
        return f'File {file_name} uploaded successfully to S3.'

    except Exception as e:
        print(f"Error uploading file {file_name}: {e}")
        return f"Error uploading {file_name}: {str(e)}"
    
# Generate presigned URL to able download files
def generate_presigned_url(bucket_name, s3_key, expiration=36000):
    try:
        print(f"\n\nGenerating presigned URL for s3 key: {s3_key}")
        result_path = 'result/'
        response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket_name, 'Key': s3_key},
                                             ExpiresIn=expiration)
        
        return response
    except Exception as e:
        print(f"Error generating presigned URL for {s3_key}: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)#, host='0.0.0.0', port=5000)
