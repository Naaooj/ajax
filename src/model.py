from tqdm import tqdm
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

import torch

class Model():
    def __init__(self, dataset, data_loader, num_epochs=5):
        self.dataset = dataset
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train_model(self):
        print('Training model...')

        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(self.data_loader) * self.num_epochs  

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(self.data_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', leave=False)

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

            avg_train_loss = total_loss / len(self.data_loader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_train_loss:.4f}')

    def evaluate(self, dataloader):
        print('Evaluating model...')

        self.model.eval()
        total_loss = 0
        correct_predictions = 0

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

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / len(dataloader.dataset)

        return avg_loss, accuracy