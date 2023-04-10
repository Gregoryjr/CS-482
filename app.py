import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
model_name = st.text_input("which model do you want to use?")
classifier = pipeline("zero-shot-classification")
res = classifier("possible lables", cata = ["offensive"],)
print(res)
st.button("submit")

