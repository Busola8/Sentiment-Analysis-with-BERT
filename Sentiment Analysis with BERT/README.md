# BERT Sentiment Analysis Project (with immediate demo model)

This project contains end-to-end scripts to fine-tune a BERT-family model for sentiment analysis.
A lightweight demo TF-IDF+LogisticRegression model is included for immediate inference without waiting for large downloads or GPU training.

## What is included
- `data/sentiment_data.csv` - sample dataset (generated)
- `src/train_hf.py` - script to fine-tune a transformer using Hugging Face Trainer (requires internet + GPU recommended)
- `src/train_fallback.py` - fast local training (TF-IDF + Logistic Regression) used to produce a demo model quickly
- `models/demo_text_model.joblib` - demo model (TF-IDF+LogReg) for quick inference
- `src/inference.py` - loads HF model if present else uses demo model
- `src/streamlit_app.py` - simple UI
- `requirements.txt` - deps

## Quickstart
1. Create virtualenv and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Run demo Streamlit app (uses included demo model):
   ```bash
   streamlit run src/streamlit_app.py
   ```
3. To fine-tune a transformer (optional):
   ```bash
   python src/train_hf.py --model_name distilbert-base-uncased --epochs 1 --batch_size 8
   ```

Note: Fine-tuning in `train_hf.py` will download models and is best on GPU. The demo model allows you to try inference immediately.
