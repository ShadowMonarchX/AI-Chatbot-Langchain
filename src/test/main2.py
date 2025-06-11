import streamlit as st
from streamlit_chat import message
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Set Streamlit page configuration
st.set_page_config(
    page_title="ChatBot.ai",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Namaste, How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Load GPT-2 model and tokenizer for query refinement and response generation
gpt2_model_name = "gpt2-medium"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_model.eval()

# Initialize Pinecone for vector-based search
pc = Pinecone(api_key='485df547-324f-4bd6-97d5-98c480140498', environment='us-east-1-aws')
index = pc.Index("chatbot2")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to refine queries using GPT-2
def query_refiner(query):
    prompt = f"Refine the following user query to be more specific:\n\nQuery: {query}\n\nRefined Query:"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=200, temperature=0.7, num_return_sequences=1)
    refined_query = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Refined Query:")[-1].strip()
    return refined_query

# Function to find the best match in Pinecone index
def find_match(input):
    input_em = model.encode(input).tolist()
    # Using keyword arguments for the query
    result = index.query(namespace="Default", vector=input_em, top_k=2, include_metadata=True)
    if result['matches']:
        # Handling the case where there may be fewer matches than requested
        matches = result['matches']
        context = ""
        for match in matches:
            if 'metadata' in match and 'text' in match['metadata']:
                context += match['metadata']['text'] + "\n"
        return context.strip()
    else:
        return "No relevant information found."
    
# Function to generate a response using GPT-2
def generate_response(context, query):
    prompt = f"Context:\n{context}\n\nUser Query:\n{query}\n\nResponse:"
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=300, temperature=0.7, num_return_sequences=1)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Response:")[-1].strip()
    return response

# Create the User Interface
st.title("ChatBot.ai 🤖")

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            refined_query = query_refiner(query)
            context = find_match(refined_query)
            response = generate_response(context, query)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
