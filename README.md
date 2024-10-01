# AJAX

## Configuring the VM
```
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
```

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
   pip install torch openai azure-cognitiveservices-vision-computervision transformers pillow fitz image tqdm scikit-learn matplotlib
   ```
4. Create a config.ini
   In the `src/` directory, create a `config.ini` in order to configure the keys, models, and api of Azure to use.
   The file should have the following content:
   ```
   [openai]
   key = your_api_key
   endpoint = your_azure_openai_endpoint
   deployment_name = your_deployment_name
   api_version = your_api_version
   ```
   Those information should NEVER be stored in git. 
5. Create a folder `resumes`
   ```
   resumes/
   ├─hired
   ├─rejected
   ```
   And put PDF files of valid or unvalid resumes accordingly
   This folder should NEVER be stored in git.

Please note that every time you want to run the project you need to activate the virtual environment. 

## Links
* https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/