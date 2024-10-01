from resume_dataset import ResumeDataset
from common.json_utils import JsonUtils

import os

class ResumeDatasetBuilder:

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hired_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/hired/')
        self.rejected_results_folder_path = os.path.join(os.getcwd(), 'resumes/results/rejected/')

    def build_dataset(self):
        texts = []
        labels = []

        print(f"Start loading hired resumes for dataset '{self.hired_results_folder_path}'")
        hired_texts, hired_labels = self.__load_data(self.hired_results_folder_path, 1)
        texts.extend(hired_texts)
        labels.extend(hired_labels)

        print(f"Start loading rejected resumes for dataset '{self.rejected_results_folder_path}'")
        rejected_texts, rejected_labels = self.__load_data(self.rejected_results_folder_path, 0)
        texts.extend(rejected_texts)
        labels.extend(rejected_labels)

        print(f"Loaded {len(texts)} resumes ({len(hired_texts)} hired and {len(rejected_texts)} rejected) for dataset")

        dataset = ResumeDataset(texts, labels, self.tokenizer, self.max_length)
    
        return dataset

    def __load_data(self, folder, label):
        texts = []
        labels = []

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    texts.append(JsonUtils.flatten(json_path))
                    labels.append(label)

        return texts, labels
