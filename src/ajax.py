from model import Model
from resume_dataset_builder import ResumeDatasetBuilder
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

import numpy as np
import os
import random
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    dataset = ResumeDatasetBuilder(tokenizer, max_length=128).build_dataset()

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_batch_size = 16
    validation_batch_size = 8
    seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator)
    validation_data_loader = DataLoader(
        validation_dataset, 
        batch_size=validation_batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=generator)
    
    # Define the directory to save the models    
    models_dir = os.path.join(os.getcwd(), 'models')

    # Train the model
    model = Model(train_data_loader, validation_data_loader, models_dir, num_epochs=10, learning_rate=1e-53, weight_decay=1e-2, patience=3)
    model.train_model()

if __name__ == '__main__':
    main()
