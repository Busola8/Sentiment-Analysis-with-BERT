# train_fallback.py - small, fast model for immediate inference (TF-IDF + LogisticRegression)
import joblib, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'sentiment_data.csv')
OUT_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'demo_text_model.joblib')

def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=['text','label'])
    df['label_num'] = df['label'].map({'negative':0,'positive':1})
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(vec.transform(X_test))
    print(classification_report(y_test, preds))
    joblib.dump({'vectorizer':vec, 'model':clf}, OUT_MODEL)
    print('Saved demo model to', OUT_MODEL)

if __name__ == '__main__':
    main()
