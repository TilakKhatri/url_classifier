import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Add the current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from feature_extraction import extract_features

def main():                      
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'sample_data', 'data.csv')

    
    df = pd.read_csv(data_path)
    urls = df['url'].tolist()
    labels = df['label'].tolist()

    X, vectorizer = extract_features(urls, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("Train accuracy:", clf.score(X_train, y_train))
    print("Test accuracy:", clf.score(X_test, y_test))

    # Save model and vectorizer in the same directory as the script
    model_path = os.path.join(script_dir, 'model.joblib')
    vectorizer_path = os.path.join(script_dir, 'vectorizer.joblib')
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved successfully")

if __name__ == "__main__":
    main()