import fitz
import os
from pinecone import Pinecone, ServerlessSpec
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import dotenv

# 環境変数など読み込み
dotenv.load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("API_KEY")
PINECONE_INDEX_NAME = "pdf-qa-bot"

# Pinecone初期化
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # Gemini埋め込みに合わせて
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Drive認証
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
info = json.loads(os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"))  # または secrets から取得
credentials = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

# Gemini埋め込み設定
genai.configure(api_key=GOOGLE_API_KEY)
embed_model = genai.GenerativeModel("embedding-001")

# PDF読み取り・ベクトル化・保存
def process_and_store_pdf(file_id, file_name):
    # PDF取得
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    doc = fitz.open(stream=fh.read(), filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # テキスト分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    # ベクトル化 & 保存
    for i, chunk in enumerate(chunks):
        embedding = embed_model.embed_content(chunk, task_type="retrieval_document")["embedding"]
        metadata = {
            "source": file_name,
            "chunk_index": i
        }
        index.upsert([(f"{file_id}_{i}", embedding, metadata)])

# 例：DriveのPDF一覧を取得して保存
folder_id = "YOUR_FOLDER_ID"
results = drive_service.files().list(q=f"'{folder_id}' in parents and mimeType='application/pdf'", fields="files(id, name)").execute()
pdf_files = results.get("files", [])

for file in pdf_files:
    process_and_store_pdf(file["id"], file["name"])
