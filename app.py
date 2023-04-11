import streamlit as st
from transformers import pipeline 
st.title("Milestone #2 v2")
text = st.text_input("write a statement")
import streamlit as st

options = ["zero-shot-classification", "cardiffnlp/twitter-roberta-base-offensive", "nill3"]
model = st.selectbox("Select an option", options)

##model = st.write("You selected:", selected_option)

con = st.button("submit")
if con:
  if model == "zero-shot-classification":
    classifier = pipeline(model)
    res = classifier(text, candidate_labels= ["offensive"],)
    print(res)
    st.write(res)
  
  if model == "cardiffnlp/twitter-roberta-base-offensive":
    

    classifier = pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-offensive', tokenizer='cardiffnlp/twitter-roberta-base-offensive')


    result = classifier(text)

    
    st.write(f"Score: {result[0]['score']*100:.2f}% confidence")
   


