import os
import json
import shutil
import hashlib
from datetime import datetime
import tempfile
import pytesseract
from docx import Document
import pdfplumber
from pdf2image import convert_from_path
import chardet
from tqdm import tqdm
import numpy as np

# Import extraction functions from document_processor
from document_processor import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt

class DocumentManager:
    def __init__(self, data_folder, metadata_file="document_metadata.json"):
        self.data_folder = data_folder
        self.metadata_file = os.path.join(data_folder, metadata_file)
        self.ensure_metadata_exists()

    def ensure_metadata_exists(self):
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({"documents": []}, f, ensure_ascii=False, indent=2)

    def load_metadata(self):
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def calculate_hash(self, file_path):
        """Считывает хеш-код содержимого файла для сравнения."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def add_document(self, file_path, title=None, description="", tags=None, is_update=False, update_doc_id=None):
        metadata = self.load_metadata()
        filename = os.path.basename(file_path)

        # text preview
        preview_text = self.extract_preview(file_path)

        # file hash
        file_hash = self.calculate_hash(file_path)

        # Проверка существует ли уже документ с таким же хешем.
        for existing_doc in metadata["documents"]:
            if existing_doc["file_hash"] == file_hash and existing_doc["status"] == "active":
                print(f"Document with the same content already exists (ID: {existing_doc['id']})")
                return existing_doc

        # уникального ID и определение версии
        new_doc = {
            "id": self.generate_id(),
            "original_filename": filename,
            "title": title or filename,
            "description": description,
            "tags": tags or [],
            "upload_date": datetime.now().isoformat(),
            "version": 1,
            "previous_version": None,
            "status": "active",
            "file_path": self.store_document(file_path),
            "file_hash": file_hash,
            "preview": preview_text[:500] if preview_text else ""
        }
        metadata["documents"].append(new_doc)
        self.save_metadata(metadata)
        return new_doc

    def store_document(self, file_path):
        """Сохраняет документ в папке данных под уникальным именем файла."""
        filename = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{filename}"
        dest_path = os.path.join(self.data_folder, new_filename)
        shutil.copy2(file_path, dest_path)
        return dest_path

    def extract_preview(self, file_path):
        """Текст превью, пока хз будем юзать или нет"""
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.docx':
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n".join(paragraphs[:5])
            elif file_ext == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    if len(pdf.pages) > 0:
                        return pdf.pages[0].extract_text() or ""
                    return ""
            elif file_ext == '.txt':
                with open(file_path, 'rb') as f:
                    raw_data = f.read(5000)
                    detected_encoding = chardet.detect(raw_data)['encoding']
                with open(file_path, 'r', encoding=detected_encoding or 'utf-8', errors='replace') as file:
                    return file.read(1000)
            return "С таким типом файлов, предварительный просмотр недоступен."
        except Exception as e:
            return f"Ошибка при выводе предварительного просмотра: {str(e)}"

    def extract_full_text(self, file_path):
        """
        Вытаскивает полный текст из документа. 
        Используется для поиска сходства.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.docx':
                return extract_text_from_docx(file_path)
            elif file_ext == '.pdf':
                text, meta = extract_text_from_pdf(file_path, min_words_per_page=50)
                return text
            elif file_ext == '.txt':
                text, meta = extract_text_from_txt(file_path)
                return text
            else:
                return ""
        except Exception as e:
            print(f"Ошибка вывода всего текста из {file_path}: {str(e)}")
            return ""

    def find_similar_documents(self, file_path, embeddings, vectorstore, top_k=5):
        """
        Учитывает путь к файлу, 
        извлекает его полный текст, 
        а после генерирует вложение и выполняет поиск по схожести в существующих документах.
        Возвращает: document_metadata, similarity_score.
        """
        try:
            text = self.extract_full_text(file_path)
            if not text.strip():
                print(f"[INFO] No text could be extracted from {file_path}")
                return []
            # Compute query embedding as a numpy array of type float32
            query_embedding = np.array(embeddings.embed_query(text[:10000]), dtype="float32")
            
            # Try calling the preferred method
            try:
                results = vectorstore.similarity_search_with_score(query_embedding, k=top_k)
            except AttributeError:
                try:
                    results = vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding, k=top_k)
                except AttributeError:
                    docs = vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
                    results = [(doc, 0.0) for doc in docs]
            
            # Map results to document metadata
            similar_docs = []
            docs_added = set()
            metadata = self.load_metadata()
            doc_map = {os.path.basename(doc["file_path"]): doc for doc in metadata["documents"]}
            for retrieved_doc, score in results:
                file_name = retrieved_doc.metadata.get("file_name")
                if file_name in doc_map and file_name not in docs_added:
                    document = doc_map[file_name]
                    similar_docs.append((document, score))
                    docs_added.add(file_name)
            return similar_docs
        except Exception as e:
            print(f"Error finding similar documents: {str(e)}")
            return []

    def delete_document_by_id(self, doc_id):
        """Отмечает документ как удаленный в метаданных (он больше не будет отображаться в активных документах)."""
        metadata = self.load_metadata()
        for doc in metadata["documents"]:
            if doc["id"] == doc_id:
                doc["status"] = "deleted"
                print(f"Document {doc['title']} (ID: {doc_id}) marked as deleted.")
        self.save_metadata(metadata)

    def generate_id(self):
        """уникальный ID документа."""
        return f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"

    def get_active_documents(self):
        """Все активные документы."""
        metadata = self.load_metadata()
        return [doc for doc in metadata["documents"] if doc["status"] == "active"]

    def get_document_by_id(self, doc_id):
        """Определенный документ по идентификатору."""
        metadata = self.load_metadata()
        for doc in metadata["documents"]:
            if doc["id"] == doc_id:
                return doc
        return None

    def initialize_from_existing_files(self):
        """Метаданные из существующих файлов в папке данных."""
        metadata = self.load_metadata()
        existing_files = set(os.path.basename(doc["file_path"]) for doc in metadata["documents"])
        for file in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.docx', '.pdf', '.txt')):
                if file not in existing_files:
                    print(f"Adding existing file to metadata: {file}")
                    self.add_document(file_path)
