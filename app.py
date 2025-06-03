import os
import io
import fitz  # PyMuPDF
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import dotenv

# --- 初期設定 ---
st.set_page_config(page_title="Drive連携PDFベクトル登録Bot", layout="wide")
dotenv.load_dotenv()

# --- 認証情報 ---
GEMINI_API_KEY = os.getenv("API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "pdf-qa-bot"
PINECONE_REGION = "us-east-1"
PINECONE_CLOUD = "aws"

# --- Google Drive認証 ---
creds = service_account.Credentials.from_service_account_info(
    {
        "type": os.getenv("type"),
        "project_id": os.getenv("project_id"),
        "private_key_id": os.getenv("private_key_id"),
        "private_key": os.getenv("private_key").replace('\\n', '\n'),
        "client_email": os.getenv("client_email"),
        "client_id": os.getenv("client_id"),
        "auth_uri": os.getenv("auth_uri"),
        "token_uri": os.getenv("token_uri"),
        "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
        "client_x509_cert_url": os.getenv("client_x509_cert_url"),
    },
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

drive_service = build("drive", "v3", credentials=creds)

# --- Gemini & Pinecone初期化 ---
genai.configure(api_key=GEMINI_API_KEY)
embed_model = genai.GenerativeModel("embedding-001")
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)

# --- ファイル一覧取得（Drive内PDF） ---
st.markdown("### 📁 Google Drive上のPDFを選択してベクトル登録")
query = "mimeType='application/pdf'"
results = drive_service.files().list(q=query, pageSize=10, fields="files(id, name)").execute()
files = results.get("files", [])

file_dict = {f["name"]: f["id"] for f in files}
selected_file = st.selectbox("PDFを選択", list(file_dict.keys()))

if st.button("📥 Driveから読み込んでベクトル登録"):
    file_id = file_dict[selected_file]

    # PDFをダウンロード
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)

    # テキスト抽出
    text = ""
    with fitz.open(stream=fh.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    st.success("📄 テキスト抽出完了")

    # チャンク分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    st.write(f"🧩 チャンク数: {len(chunks)}")

    # ベクトル化＆Pinecone保存
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embed_model.embed_content(chunk, task_type="retrieval_document")["embedding"]
        vectors.append({
            "id": f"{selected_file}_{i}",
            "values": embedding,
            "metadata": {"source": selected_file, "text": chunk}
        })

    index.upsert(vectors=vectors)
    st.success(f"✅ {selected_file} を Pinecone に登録しました")
