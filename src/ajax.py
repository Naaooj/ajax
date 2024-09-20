from model import Model
from resume_dataset_builder import ResumeDatasetBuilder
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

import os

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ResumeDatasetBuilder(tokenizer).buildDataset()

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    batch_size = 2
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # Define the directory to save the models    
    models_dir = os.path.join(os.getcwd(), 'models')

    # Train the model
    model = Model(train_data_loader, validation_data_loader, models_dir)
    model.train_model()

    # Evaluate the model
    loss, accuracy, precision, recall, f1 = model.evaluate(validation_data_loader)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Save the model
    model.save()

if __name__ == '__main__':
    main()
