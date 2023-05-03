import streamlit as st
from transformers import pipeline 

st.title("Milestone #2 offensive statement prediction with pre-trained models")
st.write("in this basic demo you can select a model to judge whether or not the text below is offensive")
text = "The mail man looks dumb"
st.write(text)

options = ["zero-shot-classification", "cardiffnlp/twitter-roberta-base-offensive", "Greys/milestonemodel"]
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
    
  if model == "Greys/milestonemodel"
  
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("Greys/milestonemodel")

    def classify_sentence(sentence):
      inputs = tokenizer(sentence, return_tensors="pt")
      outputs = model(**inputs)
      probs = outputs.logits.softmax(dim=1)
      return probs.detach().numpy()[0]
    probs = classify_sentence(text)
    print(probs)
