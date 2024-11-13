import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from io import StringIO

# Show title and description.
st.title("ChatCV")
st.write("A simple way to interact with CV")

cv = st.file_uploader(label="Upload your CV in txt format", type="txt")

if cv is not None:
    # To convert to a string based IO:
    stringio = StringIO(cv.getvalue().decode("utf-8"))

    # To read file as string:
    cv_string_data = stringio.read()
    st.write(cv_string_data)

