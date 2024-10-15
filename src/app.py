import json
import os

from flask import Flask, render_template, request, redirect

from src.classifier.classifier import ModelClassifier
from src.common.pdf_convertion import convert_pdf_to_json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)
    # get the file
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    # check if the file is allowed
    if file and allowed_file(file.filename):
        filepath = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        resume_in_json = convert_pdf_to_json(filepath)
        resume_text = json.dumps(resume_in_json)
        resume_classification = ModelClassifier().classify(resume_text)
        return "Candidate should be called in priority" if resume_classification == 1 else "Candidate should be called in normal order"
    else:
        return 'File type not allowed'


if __name__ == '__main__':
    app.run(debug=True)
