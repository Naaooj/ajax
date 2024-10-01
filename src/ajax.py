from model import Model
from resume_dataset_builder import ResumeDatasetBuilder
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

import os

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    dataset = ResumeDatasetBuilder(tokenizer, max_length=256).buildDataset()

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_batch_size = 16
    validation_batch_size = 8
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    
    # Define the directory to save the models    
    models_dir = os.path.join(os.getcwd(), 'models')

    # Train the model
    model = Model(train_data_loader, validation_data_loader, models_dir, num_epochs=6, learning_rate=1e-5, weight_decay=1e-3, patience=3)
    model.train_model()

if __name__ == '__main__':
    main()
