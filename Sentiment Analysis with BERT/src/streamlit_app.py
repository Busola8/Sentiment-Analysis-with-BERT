# streamlit_app.py
import streamlit as st
from src.inference import load_pipeline
st.title('Sentiment Analysis Demo (BERT finetune project)')
pipe = load_pipeline()
txt = st.text_area('Enter text to classify', height=200)
if st.button('Analyze'):
    res = pipe(txt)
    st.json(res)
