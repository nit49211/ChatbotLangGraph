import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Page setup
st.set_page_config(page_title="📚 Document Summarizer", layout="wide")
st.title("🤖 Document Summarizer")

# Document dropdown
doc_options = [
    os.path.relpath(os.path.join(root), "Embeddings")
    for root, _, files in os.walk("Embeddings")
    for file in files if file.endswith(".faiss")
]

doc_choice = st.sidebar.selectbox("📂 Choose a document", doc_options)

# Load vectorstore
@st.cache_resource
def load_vectorstore(doc_name):
    vector_path = os.path.join("Embeddings", *doc_name.split(os.sep))
    return FAISS.load_local(
        vector_path,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        allow_dangerous_deserialization=True
    )

# Initialize memory if not done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
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

def get_chain(vectorstore, memory):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        output_key="answer"
    )

# Load vectorstore
vectorstore = load_vectorstore(doc_choice)

# Set or update qa_chain if doc changes
if "qa_chain" not in st.session_state or st.session_state.get("loaded_doc") != doc_choice:
    st.session_state.qa_chain = get_chain(vectorstore, st.session_state.memory)
    st.session_state.loaded_doc = doc_choice

qa_chain = st.session_state.qa_chain

# Initialize chat history display
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
