import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Page setup
st.set_page_config(page_title="📚 Document Summarizer", layout="wide")
st.title("🤖 Document Summarizer")

# File uploader
uploaded_file = st.sidebar.file_uploader("📤 Upload a document", type=["pdf", "txt", "md"])

# Dropdown for existing vectorstore options
doc_options = [
    os.path.relpath(os.path.join(root), "Embeddings")
    for root, _, files in os.walk("Embeddings")
    for file in files if file.endswith(".faiss")
]
doc_choice = st.sidebar.selectbox("📂 Or choose existing document", doc_options)

# Vectorstore loader for prebuilt
@st.cache_resource
def load_vectorstore(doc_name):
    vector_path = os.path.join("Embeddings", *doc_name.split(os.sep))
    return FAISS.load_local(
        vector_path,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        allow_dangerous_deserialization=True
    )

# Handle uploaded file: embed and build FAISS
def embed_uploaded_file(upload):
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, upload.name)
    with open(file_path, "wb") as f:
        f.write(upload.getbuffer())

    if upload.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(split_docs, embeddings)

# Initialize memory if not set
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt template
QA_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Use the following conversation history and context from the document to answer the question.

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer in a clear and concise way:
""")

# Chain builder
def get_chain(vectorstore, memory):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        output_key="answer"
    )

# Use uploaded vectorstore or fallback to prebuilt
if uploaded_file:
    vectorstore = embed_uploaded_file(uploaded_file)
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.qa_chain = get_chain(vectorstore, st.session_state.memory)
    st.session_state.loaded_doc = uploaded_file.name
else:
    vectorstore = load_vectorstore(doc_choice)
    if "qa_chain" not in st.session_state or st.session_state.get("loaded_doc") != doc_choice:
        st.session_state.qa_chain = get_chain(vectorstore, st.session_state.memory)
        st.session_state.loaded_doc = doc_choice

qa_chain = st.session_state.qa_chain

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = qa_chain.invoke({"question": prompt})["answer"]
    except Exception as e:
        response = f"⚠️ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
