import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import tempfile
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="LLaMA PDF Q&A", page_icon="📄", layout="centered")

# Hide icons using only your specified CSS
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("""
    <div style="text-align:center;">
        <h1 style="color:#2E86AB;">📄 Intelligent Document Q&A</h1>
        <p style="font-size:16px;">Upload a PDF, ask questions, and get AI-powered answers based on your document.</p>
    </div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

# Initialize model with caching
@st.cache_resource(ttl=24*3600)
def load_embedding_model():
    os.makedirs("model_cache", exist_ok=True)
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        st.error("Hugging Face token not found. Please set HF_TOKEN in your environment variables.")
        st.stop()
    
    return SentenceTransformer(
        'all-MiniLM-L6-v2',
        cache_folder="model_cache",
        use_auth_token=hf_token
    )

# Initialize Groq client
def get_groq_client():
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in your environment variables.")
        st.stop()
    return Groq(api_key=groq_key)

try:
    model = load_embedding_model()
    client = get_groq_client()
except Exception as e:
    st.error(f"Failed to initialize APIs: {str(e)}")
    st.stop()

# Function to load and split PDF
def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Function to compute embeddings
def get_embeddings(documents):
    document_texts = [doc.page_content for doc in documents]
    return model.encode(document_texts, convert_to_numpy=True)

# Function to retrieve top-k documents
def retrieve_documents(query, embeddings, index, documents, k=2):
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0] if i < len(documents)]

# Function to clean response
def clean_response(response_content):
    cleaned_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return cleaned_content

# App logic
if uploaded_file is not None:
    try:
        with st.spinner("⏳ Processing document..."):
            documents = load_and_split_pdf(uploaded_file)
            embeddings = get_embeddings(documents)
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
        st.success("✅ Document processed successfully!")

        # User query input
        query = st.text_input("💬 Ask a question about the document")

        if query:
            with st.spinner("🔎 Searching for the answer..."):
                retrieved_docs = retrieve_documents(query, embeddings, index, documents)
                context = " ".join([doc.page_content for doc in retrieved_docs])

                completion = client.chat.completions.create(
                    model="deepseek-r1-distill-llama-70b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based only on the provided document."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query}"
                        }
                    ],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )

                response_content = ""
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    response_content += content

                final_answer = clean_response(response_content)

            st.markdown("---")
            st.subheader("🧠 Answer")
            st.write(final_answer)
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
else:
    st.info("📂 Please upload a PDF file to begin.")
