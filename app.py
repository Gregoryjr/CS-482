import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
text = st.text_input("write a statement")
import streamlit as st

options = ["zero-shot-classification", "nill", "nill3"]
model = st.selectbox("Select an option", options)

##model = st.write("You selected:", selected_option)

con = st.button("submit")
if con:
  classifier = pipeline(model)
  res = classifier(text, candidate_labels= ["offensive"],)
  print(res)
  st.write(res)

