from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

class Model():
    def __init__(self, train_data_loader, validation_data_loader, models_dir, num_epochs=5, learning_rate=2e-5, weight_decay=1e-2, patience=3, seed=42):
        print('Initializing model...')
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)

        if torch.cuda.is_available():
            print('model will be trained using GPU')
            self.device = torch.device('cuda')
        else:
            print('model will be trained using CPU')
            self.device = torch.device('cpu')

        # Ensure the output directory exists
        os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir

        # Set the seed for reproducibility
        self.__set_seed()

    def train_model(self):
        print('Training model...')

        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = len(self.train_data_loader) * self.num_epochs  

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # List to store the average training loss for each epoch
        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        training_precisions = []
        validation_precisions = []
        training_recalls = []
        validation_recalls = []
        training_f1s = []
        validation_f1s = []

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            # Set the model to training, enabling dropout and batch normalization layers
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            all_labels = []
            all_preds = []

            progress_bar = tqdm(self.train_data_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=False)

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Clear any previously calculated gradients before performing a backward pass
                self.model.zero_grad()

                # Perform a forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Get the loss value from the output, which is the cross-entropy loss
                loss = outputs.loss

                # Accumulate the training loss for the current batch
                total_loss += loss.item()

                # Get the predicted labels (a tensor containing the raw, unnormalized scores output by the final layer of the neural network for each class: hired or rejected)
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update the progress bar with the current loss
                progress_bar.set_postfix(loss=loss.item())

            # Print the correct predictions
            print(f'Correct predictions for epoch {epoch + 1}: {correct_predictions}')

            avg_train_loss = total_loss / len(self.train_data_loader)
            training_losses.append(avg_train_loss)

            accuracy = correct_predictions.double() / len(self.train_data_loader.dataset)
            training_accuracies.append(accuracy.cpu().numpy())

            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            training_precisions.append(precision)
            
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            training_recalls.append(recall)

            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            training_f1s.append(f1)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            # Evaluate the model on the validation set
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.evaluate(self.validation_data_loader)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy.cpu().numpy())
            validation_precisions.append(val_precision)
            validation_recalls.append(val_recall)
            validation_f1s.append(val_f1)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print('Early stopping triggered')
                break

        self.__plot_metrics(training_losses, validation_losses, training_accuracies, validation_accuracies, training_precisions, validation_precisions, training_recalls, validation_recalls, training_f1s, validation_f1s)
        self.__save_model()

    def evaluate(self, dataloader):
        print('Evaluating model...')

        # Set the model to evaluation model, disabling dropout and batch normalization layers
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        all_labels = []
        all_preds = []

        # Disable gradient calculation to save memory and speed up computations
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)

            for batch in progress_bar:
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

                # Update the progress bar with the current loss
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / len(dataloader.dataset)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return avg_loss, accuracy, precision, recall, f1
    
    def __plot_metrics(self, training_losses, validation_losses, training_accuracies, validation_accuracies, training_precisions, validation_precisions, training_recalls, validation_recalls, training_f1s, validation_f1s):
        epochs = range(1, len(training_losses) + 1)

        plt.figure(figsize=(15, 10))

        # Plot training and validation loss
        plt.subplot(2, 3, 1)
        plt.plot(epochs, training_losses, marker='o', label='Training Loss')
        plt.plot(epochs, validation_losses, marker='o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()

        # Plot training and validation accuracy
        plt.subplot(2, 3, 2)
        plt.plot(epochs, training_accuracies, marker='o', label='Training Accuracy')
        plt.plot(epochs, validation_accuracies, marker='o', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid()

        # Plot training and validation precision
        plt.subplot(2, 3, 3)
        plt.plot(epochs, training_precisions, marker='o', label='Training Precision')
        plt.plot(epochs, validation_precisions, marker='o', label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Training and Validation Precision')
        plt.legend()
        plt.grid()

        # Plot training and validation recall
        plt.subplot(2, 3, 4)
        plt.plot(epochs, training_recalls, marker='o', label='Training Recall')
        plt.plot(epochs, validation_recalls, marker='o', label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Training and Validation Recall')
        plt.legend()
        plt.grid()

        # Plot training and validation F1S
        plt.subplot(2, 3, 5)
        plt.plot(epochs, training_f1s, marker='o', label='Training F1S')
        plt.plot(epochs, validation_f1s, marker='o', label='Validation F1S')
        plt.xlabel('Epoch')
        plt.ylabel('F1S')
        plt.title('Training and Validation F1S')
        plt.legend()
        plt.grid()

        # Save the plot as an image file
        # Get the current date and time
        current_time = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        lr = f"{self.learning_rate:.0e}"
        wd = f"{self.weight_decay:.0e}"
        plot_name = f'training_{current_time}_batch_size_{self.train_data_loader.batch_size}_epochs_{self.num_epochs}_lr_{lr}_wd_{wd}.png'
        plot_path = os.path.join(self.models_dir, plot_name)
        plt.savefig(plot_path)
        print(f'Plot saved to {plot_path}')

    def __save_model(self):
        # Save the trained model to the given path
        self.model.save_pretrained(save_directory=self.models_dir, state_dict=self.model.state_dict())
        print(f'Model saved to directory {self.models_dir}')

    def __set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False