# RAG_medical
# 💊 RAG-based Paracetamol Chatbot (LINE Bot)

โปรเจกต์นี้เป็น **RAG (Retrieval-Augmented Generation)** chatbot 
ที่ทำงานผ่าน LINE โดยใช้ **LangChain + FAISS** 
และตอบคำถามจากไฟล์ PDF เกี่ยวกับ "ยาพาราเซตามอล" (`med2.pdf`)

---

## 🔧 Features
- ตอบคำถามจากเอกสารยา (RAG)
- มีคำสั่งพิเศษ:
  - `\remove` : ล้างประวัติสนทนา
  - `\end` : ปิดโปรแกรมและบันทึกประวัติ
  - `\history` : ดูประวัติการสนทนา
  - `\doc` : สรุปเนื้อหาเอกสาร
  - `\id` : แสดงรหัสนักศึกษา/ผู้ใช้
- เก็บประวัติการสนทนา (`chat_history.json`)
- ใช้ **Ollama LLM** (เช่น `llama3.1`) หรือปรับเป็นโมเดลอื่นได้

---
1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-paracetamol-linebot.git
cd rag-paracetamol-linebot

2. สร้าง virtual environment และติดตั้ง dependencies
python -m venv venv
source venv/bin/activate   # หรือ venv\Scripts\activate บน Windows
pip install -r requirements.txt

3. ตั้งค่า ENV
export LINE_CHANNEL_SECRET="YOUR_SECRET"
export LINE_CHANNEL_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
export OLLAMA_MODEL="llama3.1:latest"

4.รันเซิร์ฟเวอร์ Flask
python web_hook.py

5. Run ngrok for LINE Webhook
เปิด ngrok เพื่อ expose port 5000 (ที่ Flask ใช้):
" ngrok http 5000 "

6.
ngrok จะ generate URL ประมาณ: https://xxxx-xx-xx-xx-xx.ngrok-free.app
เเละนำไป verify ใน line developers


