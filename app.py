import streamlit as st
from transformers import pipeline 
st.title("Milestone #2")
model_name = st.text_input("which model do you want to use?")
st.button("submit")

