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
    model = Model(dataset, dataloader)
    model.train_model()

    # Evaluate the model
    loss, accuracy = model.evaluate(dataloader)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    # Create a file in the given directory to save the model
    # Construct the directory path
    models_dir = os.path.join(os.getcwd(), 'models')

    # Ensure the directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model.save(models_dir)

if __name__ == '__main__':
    main()
