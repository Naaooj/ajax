from model import Model
from resume_dataset_builder import ResumeDatasetBuilder
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import os

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ResumeDatasetBuilder(tokenizer).buildDataset()
    
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    models_dir = os.path.join(os.getcwd(), 'models')
    model = Model(dataset, dataloader, models_dir)
    model.train_model()

    # Evaluate the model
    loss, accuracy = model.evaluate(dataloader)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}') 

    # Save the model
    model.save()

if __name__ == '__main__':
    main()
