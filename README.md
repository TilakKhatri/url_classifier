# URL Classifier - Malicious URL Detection System

A comprehensive machine learning system for detecting malicious URLs to protect users from phishing attacks, malware distribution, and other cyber threats.


## Quick Start
-  (In progress Project :) with continuous improvement)
### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd url_classifier
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server**
   ```bash
   uvicorn api:app --reload
   ```

## Usage

### API Endpoints

#### URL Classification
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.google.com"}'
```

Response:
```json
{
  "label": 0,
  "probability": 0.12
}
```

#### Health Check
```bash
curl http://localhost:8000/
```

### Python Usage

<!-- #### Traditional ML Prediction
```python
from ml.predict import predict_url

url = "https://suspicious-site.com/steal-data.php"
label, probability = predict_url(url)
print(f"Prediction: {'Malicious' if label else 'Safe'} (confidence: {probability:.3f})")
```

#### Deep Learning Prediction
```python
from ml.deep_predict import predict_url_deep

# LSTM model
label, probability = predict_url_deep("https://example.com", model_type='lstm')

# BERT model
label, probability = predict_url_deep("https://example.com", model_type='bert')
```

#### Real-Time Detection
```python
from real_time_detector import RealTimeURLDetector

detector = RealTimeURLDetector(use_deep_learning=True, model_type='lstm')
result = detector.detect_url("https://suspicious-site.com")
print(f"Malicious: {result.is_malicious}, Confidence: {result.confidence:.3f}")
``` -->

## Model Training

### Traditional ML Model
```bash
cd ml
python train_classifier.py
```

<!-- ### Deep Learning Models
```python
from ml.deep_learning_models import train_deep_learning_model

# Train LSTM model
train_deep_learning_model(model_type='lstm', sample_size=10000)

# Train BERT model
train_deep_learning_model(model_type='bert', sample_size=10000)
``` -->

## Project Structure

```
url_classifier/
├── api.py                    # FastAPI server
├── real_time_detector.py     # Real-time URL monitoring
├── ml/
│   ├── predict.py           # Traditional ML prediction
│   ├── deep_predict.py      # Deep learning prediction
│   ├── deep_learning_models.py  # LSTM and BERT models
│   ├── feature_extraction.py    # URL feature extraction
│   ├── train_classifier.py      # Model training
│   ├── model.joblib            # Trained ML model
│   ├── vectorizer.joblib       # TF-IDF vectorizer
│   └── sample_data/
│       └── data.csv            # Training dataset
└── requirements.txt
```

<!-- ## Performance Metrics

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Traditional ML | 95.2% | 94.8% | 95.1% | 94.9% |
| LSTM | 96.8% | 96.5% | 96.9% | 96.7% |
| BERT | 97.1% | 97.0% | 97.2% | 97.1% | -->

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For support and questions, create an issue in the repository.

---

**Disclaimer**: This tool is for educational and research purposes. Always use in conjunction with other security measures. 

---
## Future Features
 working on these features: 
- **Multi-Model Support**: Traditional ML (Logistic Regression) and Deep Learning (LSTM, BERT) models 
- **Real-Time Detection**: Fast API endpoint for instant URL classification
- **Caching System**: Intelligent caching for improved performance
- **Batch Processing**: Support for processing multiple URLs simultaneously
- **Production Ready**: Includes logging, error handling, and monitoring capabilities
