import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()
pdf_dir = "case_prep_resources"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
all_texts = []

for file in os.listdir(pdf_dir):
    if file.lower().endswith(".pdf"):
        file_path = os.path.join(pdf_dir, file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        texts = text_splitter.split_documents(documents)
        all_texts.extend([t.page_content for t in texts])

print(f"Total chunks: {len(all_texts)}")
embeddings_array = embeddings.embed_documents(all_texts)
embeddings_array = np.array(embeddings_array)

print(f"Embeddings shape: {embeddings_array.shape}")
np.save("embeddings.npy", embeddings_array)
print("✅ Embeddings saved to embeddings.npy")


client = QdrantClient("http://localhost:6333")
if not client.collection_exists("case_prep"):
    client.create_collection(
        collection_name="case_prep",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

client.upload_collection(
    collection_name="case_prep",
    vectors=embeddings_array,
    ids=None,
    batch_size=256,
)
print("✅ Embeddings uploaded to Qdrant collection 'case_prep'")