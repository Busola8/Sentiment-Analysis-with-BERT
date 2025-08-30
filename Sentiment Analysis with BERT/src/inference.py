# inference.py - load fine-tuned transformer if available, else fallback to demo TF-IDF model
import os, joblib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'finetuned')
DEMO_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'demo_text_model.joblib')

def load_pipeline():
    # prefer HF-trained model
    if os.path.exists(HF_MODEL_DIR) and os.path.isdir(HF_MODEL_DIR) and any(os.scandir(HF_MODEL_DIR)):
        try:
            tok = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_DIR)
            return pipeline('sentiment-analysis', model=model, tokenizer=tok)
        except Exception as e:
            print('Failed to load HF model:', e)
    # fallback
    if os.path.exists(DEMO_MODEL):
        data = joblib.load(DEMO_MODEL)
        vec = data['vectorizer']
        clf = data['model']
        def predict(texts):
            X = vec.transform([texts]) if isinstance(texts, str) else vec.transform(texts)
            probs = clf.predict_proba(X).tolist()
            preds = clf.predict(X).tolist()
            out = []
            for p,prob in zip(preds,probs):
                out.append({'label':'positive' if p==1 else 'negative', 'score': max(prob)})
            return out if len(out)>1 else out[0]
        return predict
    raise RuntimeError('No model found. Train a model first.')

if __name__ == '__main__':
    pipe = load_pipeline()
    print(pipe('I love this product!'))
    print(pipe('This is terrible and I hate it.'))
