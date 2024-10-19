from torch.utils.data import Dataset

import torch

class ResumeDataset(Dataset):
    """
    The ResumeDataset class is a custom dataset class that loads the resume data 
    and tokenizes it using the provided tokenizer.

    For more information, see https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Constructor for the ResumeDataset class. All parameters are required.

        Parameters:
            texts: The list of texts
            labels: The list of labels corresponding to the texts
            tokenizer: The tokenizer to use to tokenize the texts
            max_length: The maximum length of the token ; ideally max_length should be the same as the model's max_length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Overrides the __len__ method of the Dataset class.

        Returns:
            the size of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index):
        """
        Overrides the __getitem__ method of the Dataset class.

        Returns:
            an object containing the input_ids, attention_mask, and labels
        """
        text = self.texts[index]
        label = self.labels[index]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
