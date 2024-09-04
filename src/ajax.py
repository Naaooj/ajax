from model import Model
from resume_dataset_builder import ResumeDatasetBuilder
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ResumeDatasetBuilder(tokenizer).buildDataset()
    
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model(dataset, dataloader)
    model.train_model()

if __name__ == '__main__':
    main()
