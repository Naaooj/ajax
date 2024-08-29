# AJAX

## Getting started

1. Create a virtual environment, this allows to manage dependencies for the project without affecting the system's python installation.
   ```
   python3 -m venv ajax
   ```

2. Activate the virtual environment
   ```
   source ajax/bin/activate
   ```
   
3. Install the required packages
   ```
   pip3 install torch torchtext spacy pdfminer
   ```

4. Download the spaCy model
   ```
   python3 -m spacy download en_core_web_lg
   ```

Please note that every time you want to run the project you need to activate the virtual environment. 

## Azure Cognitive

Install required libraries
```
pip install openai azure-cognitiveservices-vision-computervision pillow pymupdf
```

## Azure Storage
Install required libraries
```
pip install azure-storage-blob
```

## Phi-3-vision with LLaMA

```
pip install accelerate flash-attn transformers pytorch torchvision pytesseract bitsandbytes pillow
```

```
brew install tesseract
```