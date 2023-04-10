import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
text = st.text_input("write a statement")
con = st.button("submit")
if con:
  classifier = pipeline("zero-shot-classification")
  res = classifier(text, candidate_labels= ["offensive"],)
  print(res)
  st.write(res)

