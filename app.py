import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
text = st.text_input("write a statement")
classifier = pipeline("zero-shot-classification")
res = classifier(text, cata = ["offensive"],)
print(res)
st.button("submit")

