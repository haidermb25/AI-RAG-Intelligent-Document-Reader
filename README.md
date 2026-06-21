# AI-RAG-Intelligent-Document-Reader

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their contents. The system uses FAISS for vector search, Sentence Transformers for embeddings, and Llama models through Groq for answer generation.

## Features

* PDF upload and processing
* Semantic search using FAISS
* Sentence Transformer embeddings
* AI-powered question answering
* Streamlit web interface
* Groq Llama integration

## Technologies Used

* Python
* Streamlit
* LangChain
* Sentence Transformers
* FAISS
* Groq API
* Llama Models

## Project Structure

```text
AI-RAG-Intelligent-Document-Reader
│
├── app.py
├── .env
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/haidermb25/AI-RAG-Intelligent-Document-Reader.git
```

Move into the project directory:

```bash
cd AI-RAG-Intelligent-Document-Reader
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

## Future Improvements

* Support multiple document formats
* Chat history support
* Source citations for answers
* Multi-document retrieval
* Conversation memory

## Author

Ali Haider

* Software Engineer
* AI and Machine Learning Enthusiast

GitHub: https://github.com/haidermb25
