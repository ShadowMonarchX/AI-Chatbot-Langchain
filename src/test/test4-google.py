import os
import streamlit as st
from transformers import pipeline

# Use a more accurate model for question answering
model_name = "google/flan-t5-base"
qa_pipeline = pipeline("text2text-generation", model=model_name)

# Configure Streamlit page settings
st.set_page_config(
    page_title="ChatBot.ai",
    page_icon="",
    layout="centered"
)

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("ChatBot.ai")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask me...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Generate response using the question-answering model
    response = qa_pipeline(f"answer the question: {user_prompt}")
    assistant_response = response[0]['generated_text'].strip()
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
