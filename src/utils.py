from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import streamlit as st

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Chroma vector store from disk
persist_directory = "chroma_db"
chroma_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=chroma_embeddings)

# Find most relevant chunks from vector DB
def find_match(query, k=2):
    query_embedding = embedding_model.encode(query).tolist()
    docs = db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# Refine user query using LLM
def query_refiner(conversation, query, llm):
    prompt = f"""
    Given the following user query and conversation log, formulate a question 
    that would be the most relevant to provide the user with an answer from a knowledge base.

    CONVERSATION LOG:
    {conversation}

    Query: {query}

    Refined Query:
    """
    return llm.predict(prompt).strip()

# Build conversation string from chat history
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
