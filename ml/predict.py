import joblib
import os
import sys

# Add the current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_extraction import extract_features

def predict_url(url):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model.joblib')
    vectorizer_path = os.path.join(script_dir, 'vectorizer.joblib')

    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X, _ = extract_features([url], vectorizer=vectorizer, fit=False)

    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0][1]
    return pred, prob

if __name__ == "__main__":
    url = input("Enter a URL to classify: ")
    label, prob = predict_url(url)
    print(f"Prediction: {'Malicious' if label else 'Not Malicious'} (probability: {prob:.2f})")