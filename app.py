import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Milestone #2 offensive statement prediction with pre-trained models")
st.write("in this basic demo you can select a model to judge whether or not the text below is offensive")
text = "The mail man looks so dumb"
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
    
  if model == "Greys/milestonemodel":
  
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("Greys/milestonemodel")
    my_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  def classify_sentence(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    return probs.detach().numpy()[0]
      probs = classify_sentence(text)
  def find_largest_number(numbers):
    if not numbers:
      print("List is empty.")
      return None
      max_num = numbers[0]
      max_index = 0
      for i in range(1, len(numbers)):
        if numbers[i] > max_num:
          max_num = numbers[i]
          max_index = i
    return max_index
  print(probs)    
  index = find_largest_number(probs)
  st.write(my_list[index])
#id,toxic,severe_toxic,obscene,threat,insult,identity_hate
