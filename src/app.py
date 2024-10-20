import os

from flask import Flask, render_template, request, redirect, jsonify

from src.classifier.classifier import ModelClassifier
from src.common.json_utils import JsonUtils
from src.common.pdf_convertion import convert_pdf_to_json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../uploads'
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    # get the file
    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # check if the file is allowed
    if file and allowed_file(file.filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        resume_in_json = convert_pdf_to_json(filepath)
        resume_text = JsonUtils.flatten_content(resume_in_json)
        result = ModelClassifier().classify(resume_text)

        message = "High priority candidate" if result == 1 else "No priority detected"
        return jsonify({'message': message}), 200
    else:
        return jsonify({'message': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
