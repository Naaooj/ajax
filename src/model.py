import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

import os
import torch

class Model():
    def __init__(self, train_data_loader, validation_data_loader, models_dir, num_epochs=5, patience=3):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Ensure the directory exists
        os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir

    def train_model(self):
        print('Training model...')

        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(self.train_data_loader) * self.num_epochs  

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # List to store the average training loss for each epoch
        training_losses = []
        validation_losses = []

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(self.train_data_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.model.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update the progress bar with the current loss
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(self.train_data_loader)
            training_losses.append(avg_train_loss)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_train_loss:.4f}')

            # Evaluate the model on the validation set
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.evaluate(self.validation_data_loader)
            validation_losses.append(val_loss)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            print(f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print('Early stopping triggered')
                break
        
        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', label='Training Loss')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, marker='o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()

        # Save the plot as an image file
        plot_path = os.path.join(self.models_dir, 'training_validation_loss.png')
        plt.savefig(plot_path)
        print(f'Plot saved to {plot_path}')

    def save(self):
        # Save the trained model to the given path
        self.model.save_pretrained(save_directory=self.models_dir, state_dict=self.model.state_dict())
        print(f'Model saved to {self.models_dir}')

    def evaluate(self, dataloader):
        print('Evaluating model...')

        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / len(dataloader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, precision, recall, f1