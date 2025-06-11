import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import faiss

# Pinecone and SentenceTransformer initialization (replace with your credentials)
pc = Pinecone(api_key='485df547-324f-4bd6-97d5-98c480140498', environment='us-east-1-aws')
index = pc.Index("chatbot2")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge base data (replace with your data loading logic)
def load_data():
    data = []  # Replace with your actual data structure
    with open("../data/chatbot-dialogs.csv", "r") as f:  # Assuming CSV format
        lines = f.readlines()
        for line in lines:
            text, metadata = line.strip().split(",")
            vector = model.encode(text)
            data.append({"vector": vector.tolist(), "metadata": metadata})
    return data

data = load_data()
vectors = [d["vector"] for d in data]  # Extract vectors from data

# Create Faiss index with appropriate embedding dimension
d = faiss.IndexFlatL2(model.get_embedding_dim())
d.add(vectors)

def find_match(query):
    query_vector = model.encode(query)
    D, I = d.search(query_vector.reshape(1, -1), k=2)  # Find top k similar vectors
    return [data[i]["metadata"] for i in I.ravel()]

st.set_page_config(
    page_title="ChatBot.ai",
    page_icon="",
    layout="centered"
)

if "responses" not in st.session_state:
    st.session_state["responses"] = ["Namaste, How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

# Create the User Interface
st.title("ChatBot.ai 🤖")

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Searching..."):
            context = find_match(query)  # Use Faiss for context retrieval
            # Process context and query (logic might involve handling retrieved metadata)
            response = "Your processed response based on retrieved metadata"
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
