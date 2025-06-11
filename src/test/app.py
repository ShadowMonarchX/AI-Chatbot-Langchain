import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge base data (replace with your data loading logic)
def load_data():
    # Load data from your local storage or Pinecone
    docs = []  # Replace with logic to load documents or knowledge base
    loader = PyPDFLoader('Nepal.pdf')    
    docs = loader.load_and_split()
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(docs)
    return split_docs

split_docs = load_data()

# Generate embeddings for documents
doc_embeddings = model.encode([doc.page_content for doc in split_docs])
faiss.normalize_L2(doc_embeddings)  # Normalize embeddings

# Initialize FAISS index for cosine similarity
embedding_dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dimension)
faiss_index.add(np.array(doc_embeddings))

# Function to dynamically generate response
def generate_best_sentence_response(docs, query):
    combined_content = " ".join([doc.page_content for doc in docs])
    sentences = combined_content.split('. ')
    sentence_embeddings = model.encode(sentences)
    faiss.normalize_L2(sentence_embeddings)  # Normalize for cosine similarity
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # Normalize query
    similarities = np.dot(sentence_embeddings, query_embedding.T)
    most_relevant_idx = np.argmax(similarities)
    best_sentence = sentences[most_relevant_idx]
    if similarities[most_relevant_idx] < 0.5:
        return "I'm sorry, I couldn't find specific information related to your query."
    else:
        return f"{best_sentence.strip()}"

# Streamlit User Interface
st.set_page_config(page_title="ChatBot.ai 🤖", layout="centered")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["Namaste, How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

st.title("ChatBot.ai 🤖")

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Searching..."):
            # Find the most relevant documents from FAISS
            query_embedding = model.encode([query])
            faiss.normalize_L2(query_embedding)
            k = 2  # Number of similar documents to retrieve
            distances, indices = faiss_index.search(np.array(query_embedding), k)
            similar_docs = [split_docs[idx] for idx in indices[0]]
            
            # Generate the response
            response = generate_best_sentence_response(similar_docs, query)
            
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
