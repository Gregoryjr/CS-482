import streamlit as st
from transformers import pipeline 

st.title("Milestone #2 offensive statement prediction with pre-trained models")
st.write("in this basic demo you can select a model to judge whether or not the text below is offensive")
text = "The mail man looks super dumb"
st.write(text)

options = ["zero-shot-classification", "cardiffnlp/twitter-roberta-base-offensive"]
model = st.selectbox("Select a  pre-trained model", options)

con = st.button("Submit")
if con:
  if model == "zero-shot-classification":
    classifier = pipeline(model)
    res = classifier(text, candidate_labels=["offensive"])
    label = res['labels'][0]
    score = res['scores'][0]
    st.write(f"Prediction: {label}, Score: {score*100}% chance")
  
  if model == "cardiffnlp/twitter-roberta-base-offensive":
    classifier = pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-offensive', tokenizer='cardiffnlp/twitter-roberta-base-offensive')
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']
    st.write(f"Prediction: {label}, Score: {score*100}% chance")
