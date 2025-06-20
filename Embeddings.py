import os
import sys
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# 📝 Enter your file path here!
DATA_PATH = input("📄 Enter path to your document (.txt): ").strip()

# 📥 Load and split
loader = TextLoader(DATA_PATH, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print(f"📚 Split into {len(chunks)} chunks.")

# 🔍 Create FAISS index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 💾 Save under a dynamic folder
basename = os.path.splitext(os.path.basename(DATA_PATH))[0]
save_path = os.path.join("Embeddings", basename)
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)

# Optional: Save chunks
with open(os.path.join(save_path, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump([chunk.model_dump() for chunk in chunks], f, ensure_ascii=False, indent=2)

print(f"✅ Vector store saved to '{save_path}'")
