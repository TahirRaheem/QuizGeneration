import streamlit as st
from transformers import BertTokenizer, pipeline, T5ForConditionalGeneration, T5Tokenizer

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to split text into sentences
def split_text_into_sentences(text):
    tokens = tokenizer.tokenize(text)
    sentence = ""
    sentences = []
    for token in tokens:
        sentence += token + " "
        if token == '.':
            sentences.append(sentence.strip())
            sentence = ""
    if sentence:  # Add any remaining text as the last sentence
        sentences.append(sentence.strip())
    return sentences

# Function to generate quiz questions
def generate_quiz(text):
    sentences = split_text_into_sentences(text)
    questions_answers = []

    for sentence in sentences:
        try:
            # Summarize each sentence for possible answers
            summarized_text = summarization_pipeline(sentence)[0]['summary_text']
            
            # Generate a question from the summarized text
            question_answer = qa_pipeline({
                'question': "What is the main idea of this sentence?",
                'context': sentence
            })

            # Store the question and answer
            question = question_answer['question']
            answer = question_answer['answer']
            
            # Generate multiple-choice options
            options = [answer, summarized_text, "None of the above", "Not sure"]
            questions_answers.append((question, options, answer))
        except Exception as e:
            print(f"Error processing sentence: {sentence}")
            print(e)

    return questions_answers

# Streamlit Interface
st.title("AI-Based Quiz Generator")
input_text = st.text_area("Enter text for quiz generation", "Artificial Intelligence (AI) is transforming industries...")

if st.button("Generate Quiz"):
    quiz = generate_quiz(input_text)
    for idx, (question, options, answer) in enumerate(quiz, 1):
        st.write(f"**Question {idx}:** {question}")
        for i, option in enumerate(options):
            st.write(f"  {chr(65 + i)}) {option}")
        st.write(f"**Answer:** {answer}")
        st.write("---")
