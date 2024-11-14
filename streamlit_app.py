import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from io import StringIO

# Show title and description.
st.title("ChatCV")
st.write("A simple way to interact with CV")
# File uploader
uploaded_file = st.file_uploader(label="Upload your uploaded_file in txt format", type="txt")
# Get data from file if it's uploaded
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # Read file as string:
    uploaded_file_string_data = stringio.read()
    # st.write(uploaded_file_string_data)

# Ask a question
question = st.text_input(
    "Ask about CV",
    placeholder="Give me short summary about candidate",
    disabled=not uploaded_file,
)

if uploaded_file and question:

    question_answerer = pipeline("question-answering", model="deepset/roberta-large-squad2")

    context = "This is a CV of a candidate." + uploaded_file_string_data
    # context = uploaded_file_string_data.split(sep=".")


    basic_answer = question_answerer(question=question, context=context, min_answer_len=50)['answer']

    st.write(f"Basic answer: {basic_answer}")