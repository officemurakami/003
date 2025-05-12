import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import os
import json
import dotenv

# --- 初期設定 ---
st.set_page_config(page_title="Pinecone連携QAボット", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 環境読み込み ---
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "pdf-qa-bot"

# --- 初期化 ---
genai.configure(api_key=API_KEY)
embed_model = genai.GenerativeModel("embedding-001")
chat_model = genai.GenerativeModel("gemini-1.5-pro")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- 質問フォーム ---
with st.form("qa_form"):
    question = st.text_input("❓ 質問を入力してください", value=st.session_state.get("question", ""))
    submitted = st.form_submit_button("質問する")

if submitted and question:
    st.session_state["question"] = question

    with st.spinner("🔍 回答を生成しています..."):
        # --- 質問をベクトル化 ---
        user_embedding = embed_model.embed_content(question, task_type="retrieval_query")["embedding"]

        # --- Pinecone検索 ---
        results = index.query(vector=user_embedding, top_k=5, include_metadata=True)

        # --- 検索結果のチャンクを結合 ---
        context = ""
        for match in results["matches"]:
            meta = match["metadata"]
            source = meta.get("source", "不明ファイル")
            chunk = match.get("values") or ""
            context += f"\n\n--- {source} ---\n{chunk}"

        # --- Geminiに質問 ---
        prompt = f"""以下の社内文書を参考に質問に答えてください。

{context}

Q: {question}
"""
        response = chat_model.generate_content(prompt)

        answer = response.text if hasattr(response, "text") else response.candidates[0]['content']['parts'][0]['text']
        st.session_state["answer"] = answer

# --- 回答表示 ---
if st.session_state.get("answer"):
    st.markdown("### 回答：")
    st.write(st.session_state["answer"])

    if st.button("クリア"):
        for key in ["question", "answer"]:
            st.session_state.pop(key, None)
        st.rerun()
