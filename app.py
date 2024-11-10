import nltk

# Downloading the 'punkt' tokenizer
nltk.download('punkt')

import streamlit as st
from transformers import pipeline
import nltk
import random
import os
from nltk.data import find

# Set NLTK data path if required
nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download 'punkt' tokenizer if not present
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Load QA Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to Generate a Simple Question
def generate_simple_question(context):
    return f"What is the main point of: {context}?"

# Function to Generate Multiple Choice Options
def generate_options(question, answer, text):
    options = [answer]
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences[:5]:
        result = qa_pipeline(question=question, context=sentence)
        if result["answer"] != answer:
            options.append(result["answer"])
        if len(options) >= 4:
            break
    options = list(set(options))
    while len(options) < 4:
        options.append("Random Distractor")
    random.shuffle(options)
    return options

# Main Function to Generate Quiz
def generate_quiz(text):
    sentences = nltk.sent_tokenize(text)
    quiz = []
    for sentence in sentences:
        question = generate_simple_question(sentence)
        result = qa_pipeline(question="What is the main point?", context=sentence)
        answer = result["answer"]
        options = generate_options(question, answer, text)
        quiz.append({
            "question": question,
            "answer": answer,
            "options": options
        })
    return quiz

# Streamlit app layout
st.title("AI-Based Quiz Generator")
st.write("Enter a paragraph below to generate a quiz.")

# User input section
input_text = st.text_area("Input your text", height=200)

# Generate quiz when user inputs text
if st.button('Generate Quiz'):
    if input_text:
        quiz = generate_quiz(input_text)
        for i, qa in enumerate(quiz):
            st.write(f"**Question {i+1}:** {qa['question']}")
            for idx, option in enumerate(qa['options']):
                st.write(f"  {chr(97 + idx)}) {option}")
            st.write(f"**Answer:** {qa['answer']}")
    else:
        st.warning("Please enter a paragraph.")
