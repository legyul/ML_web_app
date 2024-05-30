import os
from clustering_main import main
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import boto3
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint
import logging

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)


# S3 Client configuration
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
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

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        s3_file_path = f"uploaded/{file.filename}"
        s3.upload_file(file_path, S3_BUCKET_NAME, s3_file_path)
        os.remove(file_path)
        return redirect(url_for('process_file', filename=file.filename))

def delete_file_from_s3(bucket_name, file_key):
    s3.delete_object(Bucket=bucket_name, Key=file_key)
    print(f"File {file_key} deleted from S3 bucket {bucket_name}")

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_file(filename):
    s3_file_path = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/uploaded/{filename}"
    
    if request.method == 'POST':
        threshold = float(request.form.get('threshold'))
        algorithm = request.form.get('algorithm')
        plot = request.form.get('plot')

        try:
            # Implement main function and generate report and result file
            pdf_file, csv_file = main(s3_file_path, threshold, algorithm, plot)

            result_folder_path = "result/"

            # Upload generated report and result file to S3 bucket
            pdf_s3_key = upload_to_s3(pdf_file, S3_BUCKET_NAME)
            csv_s3_key = upload_to_s3(csv_file, S3_BUCKET_NAME)

            # Generate presigned URL
            pdf_url = generate_presigned_url(S3_BUCKET_NAME, pdf_s3_key)
            csv_url = generate_presigned_url(S3_BUCKET_NAME, csv_s3_key)

            print(f"PDF URL: {pdf_url}")
            print(f"CSV URL: {csv_url}")

            return render_template('result.html', pdf_url=pdf_url, csv_url=csv_url)
        
        # If file extention is not suported, delet the file from S3 Bucket
        except ValueError as e:
            flash(str(e))

            delete_file_from_s3(S3_BUCKET_NAME, s3_file_path)
            return redirect(request.url)

    return render_template('process.html', filename=filename)

# Upload generated files to S3 bucket
def upload_to_s3(file_name, bucket_name):
    try:
        s3.upload_file(file_name, bucket_name, f'result/{file_name}')
        print(f"File {file_name} uploaded to S3 bucket {bucket_name} as {file_name}.\n")
        return file_name
    
    except Exception as e:
        print(f"Error uploading {file_name}: {str(e)}")
        return None

# Generate presigned URL to able download files
def generate_presigned_url(bucket_name, s3_key, expiration=36000):
    try:
        result_path = 'result/'
        response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket_name, 'Key': result_path + s3_key},
                                             ExpiresIn=expiration)
        print(f"Generated URL for {s3_key}\n")
        print(response)
        return response
    except Exception as e:
        print(f"Error generating presigned URL for {s3_key}: {str(e)}")
        return None

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     """
#     Upload a file and perform clustering analysis.
#     ---
#     parameters:
#       - name: file
#         in: formData
#         type: file
#         required: true
#         description: The file to upload.
#       - name: threshold
#         in: formData
#         type: number
#         required: true
#         description: The threshold to identify useful columns for clustering.
#       - name: algorithm
#         in: formData
#         type: string
#         required: true
#         description: The clustering algorithm to use.
#       - name: plot
#         in: formData
#         type: string
#         description: Whether to generate a plot.
#     responses:
#       302:
#         description: Redirects to the report page.
#       404:
#         description: Report not found.
#     """
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             logging.debug(f"Saving file to {file_path}")
#             file.save(file_path)
#             s3.upload_file(file_path, S3_BUCKET_NAME, filename)
#             logging.debug(f"Uploaded file {filename} to S3")

#             try:
#                 threshold = float(request.form['threshold'])
#                 algorithm = request.form['algorithm']
#                 plot = 'yes' if request.form.get('plot') == 'yes' else 'no'
#                 logging.debug(f"Parameters received - Threshold: {threshold}, Algorithm: {algorithm}, Plot: {plot}")
                
#                 # Call the main function
#                 main(filename, threshold, algorithm, plot)
#                 logging.debug("Main function executed successfully")
#                 return redirect(url_for('report'))
#             except Exception as e:
#                 logging.error(f"Error during processing: {e}")
#                 return str(e), 500

# @app.route('/report')
# def report():
#     report_path = './static/_doc/Report.pdf'
#     logging.debug(f"Checking for report at {report_path}")
#     if os.path.exists(report_path):
#         logging.debug("Report found")
#         return render_template('result.html', report_path=report_path)
#     else:
#         logging.debug("Report not found")
#         return "Report not found", 404

# @app.route('/download_report')
# def download_report():
#     report_path = './static/_doc/Report.pdf'
#     if os.path.exists(report_path):
#         return send_file(report_path, as_attachment=True)
#     else:
#         return "Report not found", 404

if __name__ == '__main__':
    app.run(debug=True)
