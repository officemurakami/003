import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import os
import dotenv

# --- 初期設定 ---
st.set_page_config(page_title="Pinecone連携QAボット", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 環境変数の読み込み (.env または secrets.toml)
dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "pdf-qa-bot"
PINECONE_REGION = "us-west-2"  # Pinecone Consoleと一致させてください

# --- Gemini 初期化 ---
genai.configure(api_key=API_KEY)
embed_model = genai.GenerativeModel("embedding-001")
chat_model = genai.GenerativeModel("gemini-1.5-pro")

# --- Pinecone 初期化 ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- インデックス自動作成（存在しない場合のみ）
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    with st.spinner("🔧 Pineconeインデックスを作成中..."):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        st.success(f"✅ インデックス `{PINECONE_INDEX_NAME}` を作成しました")

# --- インデックスへ接続
index = pc.Index(PINECONE_INDEX_NAME)

# --- Streamlit 質問フォーム
with st.form("qa_form"):
    question = st.text_input("❓ 質問を入力してください", value=st.session_state.get("question", ""))
    submitted = st.form_submit_button("質問する")

# --- 回答生成処理
if submitted and question:
    st.session_state["question"] = question
    with st.spinner("🔍 回答を生成しています..."):
        try:
            # 質問をベクトル化
            user_embedding = embed_model.embed_content(
                question,
                task_type="retrieval_query"
            )["embedding"]

            # Pinecone でベクトル検索
            results = index.query(vector=user_embedding, top_k=5, include_metadata=True)

            # コンテキスト生成
            context = ""
            for match in results["matches"]:
                meta = match["metadata"]
                source = meta.get("source", "不明ファイル")
                chunk = match.get("values") or ""
                context += f"\n\n--- {source} ---\n{chunk}"

            # Gemini に質問
            prompt = f"""以下の社内文書を参考に質問に答えてください。

{context}

Q: {question}
"""
            response = chat_model.generate_content(prompt)
            answer = response.text if hasattr(response, "text") else response.candidates[0]['content']['parts'][0]['text']
            st.session_state["answer"] = answer

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")

# --- 回答表示
if st.session_state.get("answer"):
    st.markdown("### ✅ 回答：")
    st.write(st.session_state["answer"])

    if st.button("クリア"):
        for key in ["question", "answer"]:
            st.session_state.pop(key, None)
        st.rerun()
