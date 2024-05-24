from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from clustering_main import main
import boto3
import os
from dotenv import load_dotenv

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            s3.upload_file(file_path, S3_BUCKET_NAME, filename)
            threshold = float(request.form['threshold'])
            algorithm = request.form['algorithm']
            plot = 'yes' if request.form.get('plot') == 'yes' else 'no'
            main(filename, threshold, algorithm, plot)
            return redirect(url_for('report'))

@app.route('/report')
def report():
    report_path = './static/_doc/Report.pdf'
    if os.path.exists(report_path):
        return render_template('result.html', report_path=report_path)
    else:
        return "Report not found", 404

@app.route('/download_report')
def download_report():
    report_path = './static/_doc/Report.pdf'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return "Report not found", 404

if __name__ == '__main__':
    app.run(debug=True)
