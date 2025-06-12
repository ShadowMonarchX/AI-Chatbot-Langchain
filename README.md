## 🦜 AI Chatbot with LangChain

A powerful chatbot built using **LangChain**, designed to fetch, process, and answer queries using your custom documents or internet resources. Great for Q\&A, RAG-style access, and intelligent chat functionality.

---

</br>

<h1>Screenshots:</h1>

![chatbot using lanchain output- langBot](https://github.com/user-attachments/assets/68cad26e-9235-44ed-9b73-af07838cba3c)
<p>Fig1: Full Page Chatbot Output</p>

---

### 🚀 Features

* **Document ingestion** using loaders like sitemap or directory-based URL fetchers
* **Text splitting** via `RecursiveCharacterTextSplitter`
* **Embeddings + Vector DB** with support for Weaviate (or Pinecone, Chroma, etc.)
* **Retrieval-Augmented Generation (RAG)** for accurate, context-grounded answers
* **Interactive chat** with streamlined conversations and traceable sessions

---

### 🛠️ Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/ShadowMonarchX/AI-Chatbot-Langchain.git
   cd AI-Chatbot-Langchain
   ```

2. **Setup virtual environment & install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate.bat       # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in root with:

   ```
   OPENAI_API_KEY=<your-openai-key>
   WEAVIATE_URL=<your-weaviate-endpoint>
   WEAVIATE_API_KEY=<api-key-if-needed>
   ```

4. **Ingest your documents**

   ```bash
   python ingest.py --source <docs_or_url_folder>
   ```

5. **Run the chatbot server**

   ```bash
   python chat.py
   ```

   Visit `http://localhost:8000` to start chatting.

---

### ⚙️ Architecture Overview

1. **Ingestion Module**

   * Pulls content (HTML, PDFs, or local files)
   * Splits text into manageable chunks
   * Generates embeddings using OpenAI

2. **Vector Store**

   * Stores embeddings in Weaviate (configurable for others)

3. **Chat & QA Module**

   * Converts user prompts into context-aware queries
   * Retrieves relevant documents
   * Crafts answers using LLMs with source attribution

---

### 🔧 Customization Guide

* **Change vector database**

  * Swap out Weaviate for Pinecone, Chroma, etc.
  * Update config in `config.py`

* **Add new document loaders**

  * Extend ingestion pipeline (e.g. PDFLoader, DOCXLoader)

* **Swap LLM providers**

  * Replace `OpenAI` classes with models from Anthropic, Hugging Face, etc.

* **Enable tracing & observability**

  * Integrate with LangSmith or your monitoring tool

---

### 🛡️ Production Deployment

* Containerize using Docker
* Use `Procfile` or `Makefile` for deployment scripts
* Secure with environment variables, API keys, and HTTPS
* Configure autoscaling for production traffic

---

### 📚 References and Credits

* **LangChain**: A versatile framework for building LLM-powered apps ([arxiv.org][1], [github.com][2], [en.wikipedia.org][3], [github.com][4])
* Inspired by: `chat-langchain` and RAG-style chatbot architectures

---

### 📝 License

This project is licensed under **MIT**—free and open for all uses.

---

### ❤️ Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes and push to your branch
4. Open a PR and describe your enhancements

---

### 🤝 Questions?

Feel free to open an issue or reach out via GitHub discussions.

---

This version sharpens structure, improves clarity, and follows modern best practices. Let me know if you'd like to further tailor it — such as adding installation badges, CI/CD instructions, feature roadmap, or sample screenshots!

[1]: https://arxiv.org/abs/2308.03099?utm_source=chatgpt.com "LARCH: Large Language Model-based Automatic Readme Creation with Heuristics"
[2]: https://github.com/langchain-ai/chat-langchain?utm_source=chatgpt.com "Chat-LangChain - GitHub"
[3]: https://en.wikipedia.org/wiki/LangChain?utm_source=chatgpt.com "LangChain"
[4]: https://github.com/farukalamai/ai-chatbot-using-Langchain-Pinecone/blob/main/README.md?utm_source=chatgpt.com "ai-chatbot-using-Langchain-Pinecone/README.md at main - GitHub"
