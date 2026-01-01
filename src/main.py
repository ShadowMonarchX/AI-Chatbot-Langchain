import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.chat_models import ChatOllama
from utils import get_conversation_string, query_refiner, find_match

# Streamlit app title
st.title("💬 Local FAQ ChatBot")

# Initialize chat history
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize memory for conversation
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Load local LLM from Ollama
llm = ChatOllama(model="llama3")  # or mistral, codellama, etc.

# Define prompt template
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="Answer the question as truthfully as possible using the provided context, "
             "and if the answer is not contained within the text below, say 'I don't know'."
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# Set up the conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Chat UI input/output
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Ask your question:", key="input")
    if query:
        with st.spinner("Thinking..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query, llm=llm)
            st.subheader("🔍 Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)  # from Chroma
            response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        
        # Save history
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history
with response_container:
    for i in range(len(st.session_state['responses'])):
        message(st.session_state['responses'][i], key=str(i))
        if i < len(st.session_state['requests']):
            message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
