import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a conversational model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configuring Streamlit page settings
st.set_page_config(
    page_title="DialoGPT Chat",
    page_icon="",
    layout="centered"
)

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize conversation history if not already present
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Streamlit page title
st.title("DialoGPT - ChatBot")

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

    # Prepare the input for the model
    new_input_ids = tokenizer.encode(user_prompt + tokenizer.eos_token, return_tensors='pt')

    # Add the new user input to the conversation history
    if st.session_state.chat_history_ids is None:
        # Start the conversation
        bot_input_ids = new_input_ids
    else:
        # Continue the conversation
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate response using the DialoGPT model
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids, 
        attention_mask=attention_mask,
        max_length=200, 
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True,  # Enable sampling for diverse outputs
        top_p=0.95, 
        temperature=0.75
    )

    # Decode and display the bot's response
    assistant_response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
