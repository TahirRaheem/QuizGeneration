import streamlit as st
from transformers import pipeline
import nltk
import random
from nltk.data import find

# Download required NLTK data if not present
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load QA Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to Generate a Simple Question
def generate_simple_question(context):
    return f"What is the main point of: {context}?"

# Function to Generate Multiple Choice Options
def generate_options(question, answer, text):
    options = [answer]

    # Split text into sentences for distractors
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences[:5]:  # Get a few sentences for distractors
        result = qa_pipeline(question=question, context=sentence)
        if result["answer"] != answer:
            options.append(result["answer"])
        if len(options) >= 4:  # Limit to 4 options
            break

    # Ensure there are exactly 4 options
    options = list(set(options))  # Remove duplicates
    while len(options) < 4:
        options.append("Random Distractor")  # Add random distractor
    random.shuffle(options)
    return options

# Main Function to Generate Quiz
def generate_quiz(text):
    sentences = nltk.sent_tokenize(text)  # Split the paragraph into sentences
    quiz = []

    for sentence in sentences:
        # Generate a question based on the sentence
        question = generate_simple_question(sentence)

        # Use QA to get the correct answer
        result = qa_pipeline(question="What is the main point?", context=sentence)
        answer = result["answer"]

        # Generate multiple choice options
        options = generate_options(question, answer, text)

        # Store the question and options
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
