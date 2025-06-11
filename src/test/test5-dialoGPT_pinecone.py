import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone

# Load a conversational model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load SentenceTransformer model for Pinecone
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Pinecone (outside the app for efficiency)
pc = Pinecone(api_key='485df547-324f-4bd6-97d5-98c480140498', environment='us-east-1-aws')
index = "chatbot2"  # Ensure this index exists in Pinecone

# Streamlit app configuration
st.set_page_config(
    page_title="DialoGPT Chat",
    page_icon="",
    layout="centered"
)

# Initialize chat session and history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# Display chat history
st.title("DialoGPT - ChatBot")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
user_prompt = st.chat_input("Ask me...")

if user_prompt:
    # Add user message and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Prepare input for the model
    new_input_ids = tokenizer.encode(user_prompt + tokenizer.eos_token, return_tensors='pt')

    # Add user input to conversation history
    if st.session_state.chat_history_ids is None:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)

    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate response using DialoGPT
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enable sampling for diverse outputs
        top_p=0.95,
        temperature=0.75
    )

    # Decode and display assistant response
    assistant_response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.chat_history.append({"role":  
 "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Integrate Pinecone for knowledge retrieval (optional)
    if st.checkbox("Enhance with Knowledge Base"):
        # Encode user prompt using SentenceTransformer
        user_embedding = model.encode(user_prompt)

        # Query Pinecone for similar documents (consider filtering or ranking)
        results = pc.query(index, user_embedding, metric="cosine")
        top_result = results["matches"][0]["id"]  # Access the top result (adjust as needed)

        # Retrieve relevant information from the retrieved document (optional)
        # This part would depend on how you structured your PDF data in Pinecone

        # Update assistant response based on retrieved information (optional)
        assistant_response = f"{assistant_response}\nHere's some potentially relevant information from a similar document."  # Modify based on your logic
