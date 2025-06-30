import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import joblib
import os

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer=None, max_length=128):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = str(self.urls[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            # For BERT
            encoding = self.tokenizer(
                url,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # For LSTM - character-level encoding
            chars = list(url.lower())
            char_to_idx = {char: idx for idx, char in enumerate(set(chars))}
            char_to_idx['<PAD>'] = len(char_to_idx)
            char_to_idx['<UNK>'] = len(char_to_idx)
            
            encoded = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in chars]
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            else:
                encoded += [char_to_idx['<PAD>']] * (self.max_length - len(encoded))
            
            return {
                'input_ids': torch.tensor(encoded, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class DeepLearningTrainer:
    def __init__(self, model_type='lstm', device=None):
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def prepare_data(self, urls, labels, test_size=0.2, batch_size=32):
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            urls, labels_encoded, test_size=test_size, random_state=42, stratify=labels_encoded
        )
        
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train_dataset = URLDataset(X_train, y_train, self.tokenizer)
            test_dataset = URLDataset(X_test, y_test, self.tokenizer)
        else:  # LSTM
            train_dataset = URLDataset(X_train, y_train)
            test_dataset = URLDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, label_encoder
    
    def create_model(self, vocab_size=None):
        if self.model_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=2
            )
        else:  # LSTM
            self.model = LSTMClassifier(vocab_size=vocab_size)
        
        self.model.to(self.device)
        return self.model
    
    def train(self, train_loader, test_loader, epochs=10, learning_rate=2e-5):
        if self.model_type == 'bert':
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=total_steps
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = None
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.model_type == 'bert':
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    outputs = self.model(input_ids)
                    loss = criterion(outputs, labels)
                    logits = outputs
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    if self.model_type == 'bert':
                        attention_mask = batch['attention_mask'].to(self.device)
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    else:
                        outputs = self.model(input_ids)
                        logits = outputs
                    
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {total_loss/len(train_loader):.4f}')
            print(f'Training Accuracy: {100*correct/total:.2f}%')
            print(f'Validation Accuracy: {100*val_correct/val_total:.2f}%')
            print('-' * 50)
    
    def save_model(self, model_path, tokenizer_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
        
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        if self.tokenizer and tokenizer_path:
            tokenizer_path = os.path.join(script_dir, tokenizer_path)
            self.tokenizer.save_pretrained(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path, tokenizer_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
        
        if self.model_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            if tokenizer_path:
                tokenizer_path = os.path.join(script_dir, tokenizer_path)
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            # For LSTM, you'll need to recreate the model with same architecture
            pass
        
        self.model.to(self.device)
        self.model.eval()

def train_deep_learning_model(model_type='lstm', sample_size=10000):
    """
    Train deep learning model on URL classification
    """
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'sample_data', 'data.csv')
    
    df = pd.read_csv(data_path)
    
    # Sample data for faster training (remove this for full training)
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    urls = df['url'].tolist()
    labels = df['label'].tolist()
    
    # Convert labels to binary
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Initialize trainer
    trainer = DeepLearningTrainer(model_type=model_type)
    
    # Prepare data
    train_loader, test_loader, _ = trainer.prepare_data(urls, labels_encoded)
    
    # Create and train model
    if model_type == 'lstm':
        # For LSTM, we need to determine vocab size
        all_chars = set(''.join(urls).lower())
        vocab_size = len(all_chars) + 2  # +2 for PAD and UNK tokens
        trainer.create_model(vocab_size=vocab_size)
    else:
        trainer.create_model()
    
    trainer.train(train_loader, test_loader, epochs=5)
    
    # Save model
    if model_type == 'lstm':
        trainer.save_model('lstm_model.pth')
    else:
        trainer.save_model('bert_model.pth', 'bert_tokenizer')
    
    return trainer

if __name__ == "__main__":
    # Train LSTM model
    print("Training LSTM model...")
    lstm_trainer = train_deep_learning_model(model_type='lstm', sample_size=5000)
    
    # Train BERT model (requires more memory and time)
    print("Training BERT model...")
    bert_trainer = train_deep_learning_model(model_type='bert', sample_size=2000) 