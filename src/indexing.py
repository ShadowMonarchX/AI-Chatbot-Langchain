from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os

# 1. Load documents from a directory
def load_documents(directory="data"):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} documents.")
    return documents

# 2. Split documents into chunks
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks.")
    return docs

# 3. Create vector embeddings and store in Chroma
def store_embeddings_in_chroma(docs, persist_directory="chroma_db"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    print(f"✅ Embeddings stored in {persist_directory}.")

if __name__ == "__main__":
    docs = load_documents("data")
    chunks = split_documents(docs)
    store_embeddings_in_chroma(chunks)
