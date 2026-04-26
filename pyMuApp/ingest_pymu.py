import os
import gc
from dotenv import load_dotenv
import fitz  # PyMuPDF
from tqdm import tqdm
import chromadb

load_dotenv()

# Initialize Local DB
db_client = chromadb.PersistentClient(path="./insurance_db")
collection = db_client.get_or_create_collection("health_policies")

DOCS_FOLDER = "./docs"

def ingest_docs():
    if not os.path.exists(DOCS_FOLDER):
        print(f"Directory {DOCS_FOLDER} not found.")
        return

    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    
    for file_name in tqdm(files, desc="Indexing Policies"): 
        path = os.path.join(DOCS_FOLDER, file_name)
        
        existing = collection.get(where={"source": file_name}, limit=1)
        if existing and len(existing['ids']) > 0:
            continue
            
        try:
            # LIGHTWEIGHT PARSING (PyMuPDF)
            # This reads the text natively. Extremely fast, very low RAM.
            doc = fitz.open(path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            
            # CHUNKING STRATEGY
            # Split into chunks of 2000 characters
            chunks = [full_text[i:i+2000] for i in range(0, len(full_text), 2000)]
            
            # STORING
            for i, chunk in enumerate(chunks):
                # We skip completely empty chunks to save space
                if chunk.strip():
                    collection.add(
                        documents=[chunk],
                        ids=[f"{file_name}_{i}"],
                        metadatas=[{"policy_name": file_name.lower(), "source": file_name}]
                    )
                    
            # MEMORY CLEANUP
            del full_text
            del chunks
            gc.collect() 
            
        except Exception as e:
            print(f"\nFailed on {file_name}: {e}")
            gc.collect()

if __name__ == "__main__":
    ingest_docs()