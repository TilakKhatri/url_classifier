import re
from sklearn.feature_extraction.text import TfidfVectorizer


def basic_tokenizer(url):
    url = url.lower()
    tokens = re.split(r'[\/\-\.\?\=\&\_]', url)
    tokens = [token for token in tokens if token]
    return tokens

def sanitization(url):
    tokens = basic_tokenizer(url)
    tokens = [token for token in tokens if token not in ['com', 'www']]
    return tokens

def extract_features(urls, vectorizer=None, fit=False):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(tokenizer=sanitization, token_pattern=None)
    if fit:
        X = vectorizer.fit_transform(urls)
    else:
        X = vectorizer.transform(urls)
    return X, vectorizer