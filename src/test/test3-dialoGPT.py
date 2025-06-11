import os

import streamlit as st
from transformers import pipeline

# Download pre-trained GPT-2 model (consider using a larger model for better results)
model_name = "gpt2"
generator = pipeline("text-generation", model=model_name)

# configuring streamlit page settings
st.set_page_config(
    page_title="ChatBot.ai",
    page_icon="",
    layout="centered"
)

# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# streamlit page title
st.title(" ChatBot.ai")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message
user_prompt = st.chat_input("Ask me...")

if user_prompt:
    # add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # generate response using the GPT-2 model
    response = generator(user_prompt, max_length=200, num_return_sequences=1, truncation=True)
    assistant_response = response[0]['generated_text'].strip()
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)