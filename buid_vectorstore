import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# -------------------------
# Configuration
# -------------------------
PDF_PATH = os.environ.get("RAG_PDF_PATH", r"med1.pdf")
FAISS_DIR = os.environ.get("FAISS_DIR", "./faiss_index")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# -------------------------
# Build LLM
# -------------------------
def build_chat_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

# -------------------------
# Build Prompt
# -------------------------
def build_prompt(context: str, user_input: str) -> str:
    return f"""
You are a helpful medical assistant. 
Your task: Suggest ONLY over-the-counter (OTC) medications that match the symptoms in the user question. 
Base your answer ONLY on the provided context. Do NOT use external knowledge.

Context:
{context}

Instructions:
- Always identify the symptom(s) in the question.
- Then match them with relevant medications found in the context.
- Answer in English, short and clear.
- If no medicine in the context fits the symptom, say: "Sorry, no relevant medication found in the document."

User Question:
{user_input}

Answer:
""".strip()


# -------------------------
# Build FAISS index if not exists
# -------------------------
def build_faiss_index():
    if not os.path.exists(FAISS_DIR):
        print("[INDEX] Building FAISS index from PDF...")

        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        print(f"[INDEX] Loaded {len(docs)} document(s) from PDF")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        chunks = [c for c in chunks if c.page_content.strip()]
        print(f"[INDEX] Valid chunks for FAISS: {len(chunks)}")

        if not chunks:
            raise ValueError("No valid text chunks found in PDF")

        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        vectorstore = FAISS.from_documents(chunks, embedding)
        vectorstore.save_local(FAISS_DIR)
        print(f"[INDEX] FAISS index created at {FAISS_DIR}")
    else:
        print("[INDEX] FAISS index already exists.")

# -------------------------
# Main Terminal Chat
# -------------------------
if __name__ == "__main__":
    build_faiss_index()

    print("[BOOT] Loading FAISS index…")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings=embedding, allow_dangerous_deserialization=True)

    print("[BOOT] Initializing chat LLM…")
    chat_llm = build_chat_llm()

    print("\n✅ RAG Chat is ready! Type your question (or 'exit' to quit):\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye! 👋")
            break

        start_time = time.time()

        # ใช้ MMR search เพื่อดึง context หลากหลาย
        results = vectorstore.max_marginal_relevance_search(user_input, k=10, fetch_k=40)

        # รวม context จากเอกสาร
        context = "\n\n".join(d.page_content for d in results) if results else "[No relevant information found in the document]"

        print("\n[DEBUG] Retrieved context preview:\n", context[:500], "\n---\n")

        prompt = build_prompt(context, user_input)
        response = chat_llm.invoke(prompt)
        answer = getattr(response, "content", str(response)) or "[ERROR] Empty response"

        elapsed_time = time.time() - start_time
        print(f"Bot: {answer}")
        print(f"⏱️ Response time: {elapsed_time:.2f} seconds\n")
