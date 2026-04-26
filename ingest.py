import os
import gc
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from docling.document_converter import DocumentConverter

DOCS_FOLDER = "/content/drive/MyDrive/docs"
DB_PATH = "/content/drive/MyDrive/insurance_db"

bge_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_or_create_collection(
    name="health_policies",
    embedding_function=bge_embeddings
)
converter = DocumentConverter()

def ingest_docs():
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDFs. Starting AI Extraction...")

    for file_name in tqdm(files[:100], desc="Indexing Policies"):
        path = os.path.join(DOCS_FOLDER, file_name)

        existing = collection.get(where={"source": file_name}, limit=1)
        if existing and len(existing['ids']) > 0:
            continue

        try:
            result = converter.convert(path)
            md_text = result.document.export_to_markdown()

            chunks = [md_text[i:i+2000] for i in range(0, len(md_text), 2000)]

            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    collection.add(
                        documents=[chunk],
                        ids=[f"{file_name}_{i}"],
                        metadatas=[{"policy_name": file_name.lower(), "source": file_name}]
                    )

            del result, md_text, chunks
            gc.collect()

        except Exception as e:
            print(f"\nSkipped {file_name}: {e}")
            gc.collect()

ingest_docs()