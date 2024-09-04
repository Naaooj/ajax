from torch.utils.data import DataLoader
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

import torch

class Model():
    def __init__(self, dataset, data_loader, num_epochs=5):
        self.dataset = dataset
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def train_model(self):
        device = torch.device('cpu')

        self.model.to(device)

        optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        total_steps = len(self.dataloader) * self.num_epochs  

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                self.model.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(self.dataloader)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_train_loss:.4f}')

    def evaluate(model, dataloader):
        device = torch.device('cpu')

        model.eval()
        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / len(dataloader.dataset)

        return avg_loss, accuracy