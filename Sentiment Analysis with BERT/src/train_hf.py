# train_hf.py - fine-tune a transformer model using Trainer (requires GPU for speed)
import os, argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main(model_name='distilbert-base-uncased', data_path='data/sentiment_data.csv', output_dir='models/finetuned', epochs=1, batch_size=8):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text','label'])
    df['label'] = df['label'].map({'negative':0,'positive':1})
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    ds_train = Dataset.from_pandas(train_df)
    ds_eval = Dataset.from_pandas(test_df)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    tokenized_train = ds_train.map(tokenize, batched=True)
    tokenized_eval = ds_eval.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir=os.path.join(output_dir, 'logs'),
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=2
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_eval, compute_metrics=compute_metrics, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    print('Saved model to', output_dir)

if __name__ == '__main__':
    import sys
    main()
