from src.common.json_utils import JsonUtils
from src.model_training.resume_dataset import ResumeDataset

import os

class ResumeDatasetBuilder:
    """
    The ResumeDatasetBuilder class is responsible for building the dataset representing the hired and rejected resumes.
    The created dataset is a ResumeDataset.
    """

    def __init__(self, tokenizer, max_length=512):
        """
        Constructor for the ResumeDatasetBuilder class.

        Parameters:
            tokenizer: The tokenizer to use to tokenize the texts
            max_length: The maximum length of the token, default to 512; ideally max_length should be the same as the model's max_length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.hired_results_folder_path = os.path.join(current_dir, '../resumes/results/hired/')
        self.rejected_results_folder_path = os.path.join(current_dir, '../resumes/results/rejected/')

    def build_dataset(self):
        """
        Build the dataset by loading the hired and rejected resumes and tokenizing them using the provided tokenizer.

        Returns:
            dataset: The dataset containing the tokenized resumes
        """
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
        """
        Load the resumes from the specified folder and assign the specified label to them.

        Returns:
            texts: The list of texts
            labels: The list of labels corresponding to the texts
        """
        texts = []
        labels = []

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    texts.append(JsonUtils.flatten(json_path))
                    labels.append(label)

        return texts, labels
