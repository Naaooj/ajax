# TindHire

The TindHire project is a proof of concept aims to build a machine learning model to classify resumes into two categories: "hired" and "rejected".  
The primary goal is to automate the resume screening process, helping recruiters quickly identify potential candidates.

The project involves the following key components:

1. Data Collection: Resumes are collected and stored in JSON format in two separate folders: one for hired candidates and one for rejected candidates.
2. Model Training: A machine learning model is trained using the dataset to classify resumes based on the text content.
3. Evaluation: The model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1 score.

The project leverages the power of natural language processing (NLP) and deep learning to streamline the recruitment process, making it more efficient and effective.

## Getting started

1. Create a virtual environment, this allows to manage dependencies for the project without affecting the system's python installation.
   ```
   python3 -m venv tindhire
   ```

2. Activate the virtual environment
   ```
   source tindhire/bin/activate
   ```

3. Reference the root project as PYTHONPATH
   ```
   export PYTHONPATH=. 
   ```

4. Install the required packages
   ```
   pip install torch openai azure-cognitiveservices-vision-computervision transformers pillow fitz PyMuPDF image tqdm scikit-learn matplotlib Flask
   ```

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