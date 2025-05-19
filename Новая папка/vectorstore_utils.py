import os, json, hashlib
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.docstore.document import Document as LCDocument
from data_management.document_processor import process_document_folder
# Используем MyEmbeddings из embeddings.py при инициализации в роутерах
# app one

def load_or_rebuild_vectorstore(data_folder: str, indexes_folder: str, embeddings) -> LC_FAISS:
    os.makedirs(indexes_folder, exist_ok=True)
    fp_file   = os.path.join(indexes_folder, "index_fingerprint.json")
    idx_path  = os.path.join(indexes_folder, "index.faiss")
    meta_path = os.path.join(indexes_folder, "document_metadata.json")

    # fingerprint файлов
    current = {
        os.path.join(r, f): os.path.getmtime(os.path.join(r, f))
        for r, _, fs in os.walk(data_folder)
        for f in fs if f.lower().endswith(('.docx', '.pdf', '.txt'))
    }
    fingerprint = hashlib.md5(json.dumps(current, sort_keys=True).encode()).hexdigest()

    prev = None
    if os.path.exists(fp_file):
        with open(fp_file, 'r') as f:
            try:
                prev = json.load(f)
            except:
                prev = None

    # если fingerprint совпадает, пытаемся загрузить существующий
    if prev == fingerprint and os.path.exists(idx_path) and os.path.exists(meta_path):
        try:
            vs = LC_FAISS.load_local(indexes_folder, embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception:
            # при ошибке удаляем сломанные файлы и пойдём на rebuild
            for p in [idx_path, meta_path]:
                if os.path.exists(p): os.remove(p)

    # rebuild index из документов
    chunks = process_document_folder(
        data_folder,
        min_words_per_page=30,
        target_chunk_size=512,
        min_chunk_size=256,
        overlap_size=200,
        include_metadata=True
    )
    docs = [LCDocument(page_content=c['text'], metadata=c['metadata']) for c in chunks]
    vs = LC_FAISS.from_documents(docs, embeddings)
    vs.save_local(indexes_folder)
    with open(fp_file, 'w') as f:
        json.dump(fingerprint, f)
    return vs