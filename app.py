import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
text = st.text_input("write a statement")
model = st.text_input("pick a model to check the the above statement is offensive")
con = st.button("submit")
if con:
  classifier = pipeline(model)
  res = classifier(text, candidate_labels= ["offensive"],)
  print(res)
  st.write(res)

