import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ModelClassifier:

    def __init__(self):
        # Use a pre-trained tokenizer that matches your model type
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self.model = AutoModelForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), 'models/'))
        self.max_length = 128
        self.model.eval()

    def classify(self, text):
        # Tokenize the input text
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

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_class = torch.max(logits, dim=1)

        return predicted_class.item()


# def classify_folder(folder):
#     for root, _, files in os.walk(folder):
#         total = len(files)
#         hired = 0
#         for file in files:
#             text = JsonUtils.flatten(os.path.join(root, file))
#             predicted_class = classifier.classify(text)
#             hired += predicted_class
#         print(f'Folder "{folder}" Hired: {hired}/{total}')
#
#
# if __name__ == '__main__':
#     model_dir = os.path.join(os.getcwd(), 'models/')
#     max_length = 128
#     classifier = ModelClassifier(model_dir, max_length)
#     classify_folder(os.path.join(os.getcwd(), 'resumes/results/hired/'))
#     classify_folder(os.path.join(os.getcwd(), 'resumes/results/rejected/'))
