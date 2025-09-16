import os
import time
import json
import atexit
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -------------------------
# Configuration
# -------------------------
PDF_PATH = os.environ.get("RAG_PDF_PATH", r"med2.pdf")
FAISS_DIR = os.environ.get("FAISS_DIR", "./faiss_index")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "3"))

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "XXXXXXXXXXXX")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "XXXXXXXXXXXXXXXXXX")

# -------------------------
# Global Chat History
# -------------------------
chat_history = []
MAX_HISTORY = 4
CHAT_HISTORY_FILE = "chat_history.json"

# -------------------------
# Build LLM
# -------------------------
def build_chat_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

# -------------------------
# Build Prompt
# -------------------------
def build_prompt(context: str, user_input: str) -> str:
    return f"""
You are a helpful assistant. Use ONLY the following context to answer the user's question. 
Do NOT use external knowledge.

Context:
{context}

Instructions:
- Analyze your answer in Thai.
- give more information
- Ensure correct spelling
- If the context does not have enough information, say: "Sorry, I don't have enough information.

User Question:
{user_input}

Answer:
""".strip()

# -------------------------
# Build FAISS index
# -------------------------
def build_faiss_index():
    if not os.path.exists(FAISS_DIR):
        print("[INDEX] Building FAISS index from PDF...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        print(f"[INDEX] Loaded {len(docs)} document(s) from PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=250,
            chunk_overlap=70,
        )

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
# Save Chat History on Exit
# -------------------------
def save_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Chat history saved to {CHAT_HISTORY_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save chat history: {e}")

atexit.register(save_chat_history)

# -------------------------
# Flask + LINE Bot
# -------------------------
app = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    start_time = time.time()
    query = event.message.text.strip()

    # -------------------------
    # Special commands
    # -------------------------
    if query == "\\remove":
        chat_history.clear()
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🧹 ล้างประวัติการสนทนาเรียบร้อยแล้ว")
        )
        return

    if query == "\\end":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="👋 จบโปรแกรมและบันทึกประวัติเรียบร้อยแล้ว")
        )
        save_chat_history()  # เซฟประวัติอีกรอบก่อนออก
        os._exit(0)  # จบโปรแกรม

    if query == "\\history":
        if not chat_history:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="📜 ยังไม่มีประวัติการสนทนา")
            )
        else:
            # แปลงประวัติเป็นข้อความสั้น ๆ
            history_texts = []
            for role, msg in chat_history:
                prefix = "👤" if role == "User" else "🤖"
                history_texts.append(f"{prefix} {msg}")
            
            history_output = "\n".join(history_texts)
            if len(history_output) > 1900:  # กันข้อความยาวเกิน limit LINE
                history_output = history_output[:1900] + "\n… (truncated)"

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"📜 ประวัติการสนทนา:\n\n{history_output}")
            )
        return


    if query == "\\id":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="้my id is 6610110546")
        )
        return

    if query == "\\doc":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=""" เนื้อหาในเอกสารมีดังนี้:
1. คุณสมบัติของยา: อธิบายสรรพคุณ, กลไกการออกฤทธิ์, การดูดซึม และการขับยาออกจากร่างกาย
2. ผลข้างเคียงและความเสี่ยง: กล่าวถึงผลข้างเคียงที่อันตราย โดยเฉพาะพิษต่อตับ และข้อควรระวังในการใช้ยาสำหรับผู้ที่มีความเสี่ยงสูง
3. ปฏิกิริยาระหว่างยา: อธิบายปฏิกิริยาของยากับยาชนิดอื่นและแอลกอฮอล์ที่อาจเพิ่มความเสี่ยงต่ออันตราย""")
        )
        return

    
    # -------------------------
    # ปกติทำ RAG + LLM
    # -------------------------
    results = app.config["VECTORSTORE"].similarity_search_with_score(query, k=RETRIEVAL_K)
    filtered_docs = [doc for doc, score in results if score >= 0.65]

    context = "\n\n".join(d.page_content for d in filtered_docs) if filtered_docs else "[No relevant information found in the document]"

    prompt = build_prompt(context, query)
    response = app.config["CHAT_LLM"].invoke(prompt)
    answer = getattr(response, "content", str(response)) or "[ERROR] Empty response"

    # -------------------------
    # Manage chat history (max 4)
    # -------------------------
    chat_history.append(("User", query))
    chat_history.append(("Bot", answer))

    while len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)
        chat_history.pop(0)

    elapsed_time = time.time() - start_time
    answer_with_time = f"{answer}\n\n⏱️ Response time: {elapsed_time:.2f} seconds"

    if len(answer_with_time) > 1900:
        answer_with_time = answer_with_time[:1900] + "\n… (truncated)"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer_with_time))


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    build_faiss_index()

    print("[BOOT] Loading FAISS index…")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings=embedding, allow_dangerous_deserialization=True)

    print("[BOOT] Initializing chat LLM…")
    chat_llm = build_chat_llm()

    app.config["VECTORSTORE"] = vectorstore
    app.config["CHAT_LLM"] = chat_llm

    port = int(os.environ.get("PORT", "5000"))
    print(f"[RUN] Flask listening on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
