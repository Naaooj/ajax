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
   pip install openai azure-cognitiveservices-vision-computervision pillow pymupdf
   ```
4. Create a config.ini
   In the `src/` directory, create a `config.ini` in order to configure the keys, models, and api of Azure to use.
   Those information should NEVER be stored in git. 
   The file should have the following content:
   ```
   [openai]
   key = your_api_key
   endpoint = your_azure_openai_endpoint
   deployment_name = your_deployment_name
   api_version = your_api_version
   ```

Please note that every time you want to run the project you need to activate the virtual environment. 
