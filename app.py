from flask import Flask, render_template, request, send_file
from clustering_main import main
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        file_key = request.form['file_key']
        threshold = float(request.form['threshold'])
        algorithm = request.form['algorithm']
        plot = request.form.get('yes', 'no')  # Plot option default is 'no'

        # Run clustering and generate report
        main(file_key, threshold, algorithm, plot)

        # Return result page
        return render_template('result.html')

@app.route('/download_report')
def download_report():
    report_path = './static/_doc/Report.pdf'
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return "Report not found", 404

if __name__ == '__main__':
    app.run(debug=True)
