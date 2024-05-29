import os
from clustering_main import main
from flask import Flask, render_template, request, redirect, url_for, send_file
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
        s3.upload_file(file_path, S3_BUCKET_NAME, file.filename)
        os.remove(file_path)  # 업로드 후에 로컬 파일 삭제
        return redirect(url_for('process_file', filename=file.filename))

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process_file(filename):
    s3_file_path = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    if request.method == 'POST':
        threshold = float(request.form.get('threshold'))
        algorithm = request.form.get('algorithm')
        plot = request.form.get('plot')

        pdf_path, csv_path = main(s3_file_path, threshold, algorithm, plot)

        return render_template('result.html', pdf_path=pdf_path, csv_path=csv_path)

    return render_template('process.html', filename=filename)
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
