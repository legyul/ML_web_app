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

template_dir = os.path.abspath('./templates')
app = Flask(__name__, template_folder=template_dir)

#print("Template directory: ", os.path.abspath('./templates'))
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
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    task = request.form.get('task')
    print(task)
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        s3_file_path = f"uploaded/{file.filename}"
        s3.upload_file(file_path, S3_BUCKET_NAME, s3_file_path)
        os.remove(file_path)
        
        # User choose clustering option
        if task == 'clustering':
            print("clustering chose")
            return redirect(url_for('process_clustering', filename=file.filename))

        
        # User choose classification option
       # elif task == 'classification':
       #     return redirect(url_for('process_classification', filename=file.filename))
    
    return redirect(url_for('index'))


def delete_file_from_s3(bucket_name, file_key):
    s3.delete_object(Bucket=bucket_name, Key=file_key)
    print(f"File {file_key} deleted from S3 bucket {bucket_name}")

@app.route('/process_clustering/<filename>', methods=['GET', 'POST'])
def process_clustering(filename):
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

    return render_template('process_clustering.html', filename=filename)

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
        # print(f"Generated URL for {s3_key}\n")
        # print(response)
        return response
    except Exception as e:
        print(f"Error generating presigned URL for {s3_key}: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=False)


# import os
# from clustering_main import main
# from flask import Flask, render_template, request, redirect, url_for, send_file, flash
# from werkzeug.utils import secure_filename
# import boto3
# from dotenv import load_dotenv
# from flask_swagger_ui import get_swaggerui_blueprint
# import logging
# from multiprocessing import Process, Queue
# import uuid

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)

# # S3 Client configuration
# s3 = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
# )

# S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
# UPLOAD_FOLDER = '/tmp'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# SWAGGER_URL = '/swagger'
# API_URL = '/static/swagger.json'
# swagger_ui_blueprint = get_swaggerui_blueprint(
#     SWAGGER_URL,
#     API_URL,
#     config={
#         'app_name': "ML Platform"
#     }
# )

# app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/clustering')
# def clustering():
#     return render_template('upload_clustering.html')

# @app.route('/classification')
# def classification():
#     return render_template('upload_classification.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     task = request.form.get('task')
#     print(task)
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
#         file.save(file_path)
        
#         s3_file_path = f"uploaded/{file.filename}"
#         s3.upload_file(file_path, S3_BUCKET_NAME, s3_file_path)
#         os.remove(file_path)
        
#         # User choose clustering option
#         if task == 'clustering':
#             print("clustering chose")
#             return redirect(url_for('process_clustering', filename=file.filename))

#         # User choose classification option
#         # elif task == 'classification':
#         #     return redirect(url_for('process_classification', filename=file.filename))
    
#     return redirect(url_for('index'))


# def delete_file_from_s3(bucket_name, file_key):
#     s3.delete_object(Bucket=bucket_name, Key=file_key)
#     print(f"File {file_key} deleted from S3 bucket {bucket_name}")

# def perform_clustering(s3_file_path, threshold, algorithm, plot, result_queue):
#     try:
#         # Implement main function and generate report and result file
#         pdf_file, csv_file = main(s3_file_path, threshold, algorithm, plot)
#         print("perform clustering")
#         result_queue.put((pdf_file, csv_file))
#     except Exception as e:
#         result_queue.put(e)

# @app.route('/process_clustering/<filename>', methods=['GET', 'POST'])
# def process_clustering(filename):
#     s3_file_path = f"uploaded/{filename}"
#     local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     s3.download_file(S3_BUCKET_NAME, s3_file_path, local_file_path)
    
#     if request.method == 'POST':
#         threshold = float(request.form.get('threshold'))
#         algorithm = request.form.get('algorithm')
#         plot = request.form.get('plot')

#         result_queue = Queue()
#         process = Process(target=perform_clustering, args=(local_file_path, threshold, algorithm, plot, result_queue))
#         process.start()
#         print("process is started")
#         process.join()

#         result = result_queue.get()
#         if isinstance(result, Exception):
#             flash(str(result))
#             print("flash")
#             delete_file_from_s3(S3_BUCKET_NAME, s3_file_path)
#             return redirect(request.url)

#         pdf_file, csv_file = result

#         result_folder_path = "result/"
#         print("process is ended")

#         # Upload generated report and result file to S3 bucket
#         pdf_s3_key = upload_to_s3(pdf_file, S3_BUCKET_NAME, result_folder_path)
#         csv_s3_key = upload_to_s3(csv_file, S3_BUCKET_NAME, result_folder_path)

#         # Generate presigned URL
#         pdf_url = generate_presigned_url(S3_BUCKET_NAME, pdf_s3_key)
#         csv_url = generate_presigned_url(S3_BUCKET_NAME, csv_s3_key)

#         print(f"PDF URL: {pdf_url}")
#         print(f"CSV URL: {csv_url}")

#         return render_template('result.html', pdf_url=pdf_url, csv_url=csv_url)

#     return render_template('process_clustering.html', filename=filename)

# # Upload generated files to S3 bucket
# def upload_to_s3(file_name, bucket_name, folder_path):
#     try:
#         s3_key = folder_path + file_name
#         s3.upload_file(file_name, bucket_name, s3_key)
#         print(f"File {file_name} uploaded to S3 bucket {bucket_name} as {s3_key}.\n")
#         return s3_key
    
#     except Exception as e:
#         print(f"Error uploading {file_name}: {str(e)}")
#         return None

# # Generate presigned URL to able download files
# def generate_presigned_url(bucket_name, s3_key, expiration=36000):
#     try:
#         response = s3.generate_presigned_url('get_object',
#                                              Params={'Bucket': bucket_name, 'Key': s3_key},
#                                              ExpiresIn=expiration)
#         # print(f"Generated URL for {s3_key}\n")
#         # print(response)
#         return response
#     except Exception as e:
#         print(f"Error generating presigned URL for {s3_key}: {str(e)}")
#         return None

# if __name__ == '__main__':
#     app.run(debug=True)
