import os
import json
import hashlib
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.docstore.document import Document as LCDocument, Document
from data_management.document_processor import process_document_folder
from app.embeddings import embeddings

def load_or_rebuild_vectorstore(data_folder: str, indexes_folder: str, call_id: str = "") -> LC_FAISS:
    os.makedirs(indexes_folder, exist_ok=True)
    fingerprint_file = os.path.join(indexes_folder, "index_fingerprint.json")
    index_path = os.path.join(indexes_folder, "index.faiss")
    metadata_path = os.path.join(indexes_folder, "document_metadata.json")

    current_fingerprint = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith((".docx", ".pdf", ".txt")):
                path = os.path.join(root, file)
                current_fingerprint[path] = os.path.getmtime(path)
    fingerprint_hash = hashlib.md5(json.dumps(current_fingerprint, sort_keys=True).encode()).hexdigest()

    previous_fingerprint = None
    if os.path.exists(fingerprint_file):
        try:
            with open(fingerprint_file, 'r') as f:
                previous_fingerprint = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ERROR] Could not read fingerprint file: {e}")

    print(f"[DEBUG] Current fingerprint: {fingerprint_hash}")
    print(f"[DEBUG] Previous fingerprint: {previous_fingerprint}")

    if (os.path.exists(index_path) and os.path.exists(metadata_path) and previous_fingerprint == fingerprint_hash):
        print(f"[DEBUG] Index file exists: {os.path.exists(index_path)}")
        print(f"[DEBUG] Metadata file exists: {os.path.exists(metadata_path)}")
        try:
            try:
                vectorstore = LC_FAISS.load_local(indexes_folder, embeddings, allow_dangerous_deserialization=True)
                print(f"[INFO] Loaded existing vectorstore with {vectorstore.index.ntotal} vectors")
                return vectorstore
            except (AttributeError, ValueError, FileNotFoundError) as e:
                print(f"[ERROR] Failed to load existing vectorstore (specific error): {e}")
                if os.path.exists(index_path):
                    print(f"[DEBUG] Removing corrupted index: {index_path}")
                    os.remove(index_path)
                if os.path.exists(metadata_path):
                    print(f"[DEBUG] Removing corrupted metadata: {metadata_path}")
                    os.remove(metadata_path)
        except Exception as e:
            print(f"[ERROR] Unexpected error loading vectorstore: {e}")
            if os.path.exists(index_path):
                print(f"[DEBUG] Removing corrupted index: {index_path}")
                os.remove(index_path)
            if os.path.exists(metadata_path):
                print(f"[DEBUG] Removing corrupted metadata: {metadata_path}")
                os.remove(metadata_path)

    print("[INFO] Building/rebuilding index from documents...")
    chunks = process_document_folder(
        data_folder,
        min_words_per_page=100,
        target_chunk_size=512,
        min_chunk_size=256,
        overlap_size=150
    )
    print(f"[DEBUG] Generated {len(chunks)} chunks")

    if not chunks:
        try:
            vectorstore = LC_FAISS.from_documents([Document(page_content="Empty index", metadata={})], embeddings)
            vectorstore.save_local(indexes_folder)
            with open(fingerprint_file, 'w') as f:
                json.dump(fingerprint_hash, f)
            print("[WARNING] No chunks generated, created empty index")
        except Exception as e:
            print(f"[ERROR] Failed to create empty vectorstore: {e}")
            raise
        return vectorstore

    try:
        docs = [LCDocument(page_content=ch["text"], metadata=ch["metadata"]) for ch in chunks]
        vectorstore = LC_FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(indexes_folder)
        with open(fingerprint_file, 'w') as f:
            json.dump(fingerprint_hash, f)
        print(f"[INFO] Created new vectorstore with {vectorstore.index.ntotal} vectors")
    except Exception as e:
        print(f"[ERROR] Failed to create vectorstore: {e}")
        raise
    return vectorstore
