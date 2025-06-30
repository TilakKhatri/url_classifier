import torch
import torch.nn as nn
import numpy as np
import os
from deep_learning_models import LSTMClassifier
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

class DeepURLPredictor:
    def __init__(self, model_type='lstm', model_path=None, tokenizer_path=None):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        
        if model_path:
            self.load_model(model_path, tokenizer_path)
    
    def load_model(self, model_path, tokenizer_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
        
        if self.model_type == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            if tokenizer_path:
                tokenizer_path = os.path.join(script_dir, tokenizer_path)
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:  # LSTM
            # For LSTM, we need to recreate the model with same architecture
            # This is a simplified version - in production, save the model architecture too
            vocab_size = 100  # This should match your training vocab size
            self.model = LSTMClassifier(vocab_size=vocab_size)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_url(self, url, max_length=128):
        if self.model_type == 'bert':
            if not self.tokenizer:
                raise ValueError("Tokenizer not loaded for BERT model")
            
            encoding = self.tokenizer(
                url,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            return encoding
        else:
            # LSTM character-level encoding
            chars = list(url.lower())
            # Create a simple char mapping (in production, use the same mapping as training)
            char_to_idx = {char: idx for idx, char in enumerate(set(chars))}
            char_to_idx['<PAD>'] = len(char_to_idx)
            char_to_idx['<UNK>'] = len(char_to_idx)
            
            encoded = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in chars]
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded += [char_to_idx['<PAD>']] * (max_length - len(encoded))
            
            return {'input_ids': torch.tensor([encoded], dtype=torch.long)}
    
    def predict(self, url):
        """
        Predict if URL is malicious
        Returns: (prediction, probability)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Preprocess URL
        inputs = self.preprocess_url(url)
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if self.model_type == 'bert':
                outputs = self.model(**inputs)
                logits = outputs.logits
            else:
                outputs = self.model(inputs['input_ids'])
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            probability = probabilities[0][1].item()  # Probability of malicious
        
        return prediction, probability
    
    def predict_batch(self, urls):
        """
        Predict multiple URLs at once
        """
        results = []
        for url in urls:
            pred, prob = self.predict(url)
            results.append((pred, prob))
        return results

def predict_url_deep(url, model_type='lstm'):
    """
    Convenience function for single URL prediction
    """
    if model_type == 'lstm':
        model_path = 'lstm_model.pth'
        tokenizer_path = None
    else:
        model_path = 'bert_model.pth'
        tokenizer_path = 'bert_tokenizer'
    
    predictor = DeepURLPredictor(model_type=model_type, model_path=model_path, tokenizer_path=tokenizer_path)
    return predictor.predict(url)

if __name__ == "__main__":
    # Test prediction
    test_urls = [
        "https://www.google.com",
        "http://malicious-site.com/steal-data.php",
        "https://github.com",
        "http://suspicious-domain.net/popup.exe"
    ]
    
    print("Testing LSTM model...")
    try:
        lstm_predictor = DeepURLPredictor(model_type='lstm', model_path='lstm_model.pth')
        for url in test_urls:
            pred, prob = lstm_predictor.predict(url)
            print(f"URL: {url}")
            print(f"Prediction: {'Malicious' if pred else 'Safe'} (confidence: {prob:.3f})")
            print("-" * 50)
    except Exception as e:
        print(f"LSTM model not available: {e}")
    
    print("\nTesting BERT model...")
    try:
        bert_predictor = DeepURLPredictor(model_type='bert', model_path='bert_model.pth', tokenizer_path='bert_tokenizer')
        for url in test_urls:
            pred, prob = bert_predictor.predict(url)
            print(f"URL: {url}")
            print(f"Prediction: {'Malicious' if pred else 'Safe'} (confidence: {prob:.3f})")
            print("-" * 50)
    except Exception as e:
        print(f"BERT model not available: {e}") 