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

3. Reference the root project as PYTHONPATH
   ```
   export PYTHONPATH=. 
   ```

4. Install the required packages
   ```
   pip install torch openai azure-cognitiveservices-vision-computervision transformers pillow fitz image tqdm scikit-learn matplotlib
   ```

5. Create a config.ini
   In the `src/common` directory, create a `config.ini` in order to configure the keys, models, and api of Azure to use.
   The file should have the following content:
   ```
   [openai]
   key = your_api_key
   endpoint = your_azure_openai_endpoint
   deployment_name = your_deployment_name
   api_version = your_api_version
   ```
   Those information should NEVER be stored in git. 

6. Create a folder `src/resumes`
   ```
   resumes/
   ├─hired
   ├─rejected
   ```
   And put PDF files of valid or unvalid resumes accordingly
   This folder should NEVER be stored in git.

Please note that every time you want to run the project you need to activate the virtual environment and reference the python path.

### Creating the dataset
See [the corresponding documentation](src/dataset_builder/README.md)

### Training the model
See [the corresponding documentation](src/model_training/README.md)  

### Utils  
See [the corresponding documentation](src/utils/README.md)  

## Configuring the VM for faster training with NVIDIA GPU
Make sure the OS is up to date:
```
sudo apt update
sudo apt upgrade
```

Add the NVIDIA package repository
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

Identify the recommended GPU:
```
ubuntu-drivers devices
```

Install the drivers matching the recommended version:
```
sudo apt install nvidia-driver-XXX
```

Reboot the machine:
```
sudo reboot
```

Check the status of the graphic card:
```
nvidia-smi
```

Install CUDA toolkit:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

Install python:
```
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev
```

## Links
* https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
* https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75