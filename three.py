import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel  # 统一使用BERT组件
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import os

model_name = 'bert-base-chinese'

class RumorDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length, image_dir=None):
        self.data = pd.read_csv(
            filename, 
            sep='\t', 
            header=0,
            names=['label', 'text_a'],
            dtype={'label': int}
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.eval()
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text_a']
        label = int(self.data.iloc[idx]['label'])
        
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        image_feature = torch.zeros(512)
        if self.image_dir:
            image_path = os.path.join(self.image_dir, f"{idx}.jpg")
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                    with torch.no_grad():
                        image_feature = self.resnet(image.unsqueeze(0)).squeeze()
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'image_feature': image_feature,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalClassifier(nn.Module):
    def __init__(self, text_feature_dim, image_feature_dim, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)  # 改为BertModel
        self.classifier = nn.Linear(text_feature_dim + image_feature_dim, num_labels)

    def forward(self, input_ids, attention_mask, image_feature):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((text_features, image_feature), dim=1)
        return self.classifier(combined)

def train_model(train_loader, model, epochs=3, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_feature = batch['image_feature'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image_feature)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(test_loader, model):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_feature = batch['image_feature'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, image_feature)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            predictions.extend(preds.cpu().tolist())
            actuals.extend(labels.cpu().tolist())
    accuracy = correct / total
    return predictions, actuals, accuracy

def main():
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    TEXT_FEATURE_DIM = 768
    IMAGE_FEATURE_DIM = 512
    NUM_LABELS = 2
    train_path = 'data_test/train.tsv'
    test_path = 'data_test/torch.txt'
    image_path = 'data_test/images'

    tokenizer = BertTokenizer.from_pretrained(model_name)  # 改为BertTokenizer
    train_dataset = RumorDataset(train_path, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalClassifier(TEXT_FEATURE_DIM, IMAGE_FEATURE_DIM, NUM_LABELS)
    train_model(train_loader, model, EPOCHS)

    test_dataset = RumorDataset(test_path, tokenizer, MAX_LENGTH, image_dir = image_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    predictions, actuals, accuracy = evaluate_model(test_loader, model)

    print("预测结果\t实际标签")
    for pred, actual in zip(predictions, actuals):
        print(f"{pred}\t\t{actual}")
    print(f"正确率: {accuracy:.4f}")

if __name__ == '__main__':
    main()