import os
import json
import hashlib
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from dotenv import load_dotenv

# Импортируем модули проекта
from document_manager import DocumentManager
from document_processor import process_document_folder
from vector_storage import create_faiss_index, load_faiss_index, save_faiss_index, add_chunks_to_index

# Импорт компонентов LangChain и OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain.docstore.document import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.base import Embeddings

# ЗАГРУЗКА ENV
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Для учителей
DATA_FOLDER_TEACHERS = os.getenv("DATA_FOLDER", "data")
INDEXES_FOLDER_TEACHERS = os.getenv("INDEXES_FOLDER", "indexes")
# Для студентов
DATA_FOLDER_STUDENTS = os.getenv("DATA_FOLDER_STUD", "data_stud")
INDEXES_FOLDER_STUDENTS = os.getenv("INDEXES_FOLDER_STUD", "indexes_stud")

# ============================================================
# Класс эмбеддингов (на базе sentence_transformers)
# ============================================================
class MyEmbeddings(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

embeddings = MyEmbeddings()

# ============================================================
# Функция загрузки/перестроения векторного хранилища
# (упрощённая версия с контрольной меткой (fingerprint))
# ============================================================
def load_or_rebuild_vectorstore(data_folder: str, indexes_folder: str) -> LC_FAISS:
    fingerprint_file = os.path.join(indexes_folder, "index_fingerprint.json")
    current_fingerprint = {}
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf', '.txt')):
                path = os.path.join(root, file)
                current_fingerprint[path] = os.path.getmtime(path)
    fingerprint_hash = hashlib.md5(json.dumps(current_fingerprint, sort_keys=True).encode()).hexdigest()
    
    previous_fingerprint = None
    if os.path.exists(fingerprint_file):
        try:
            with open(fingerprint_file, 'r') as f:
                previous_fingerprint = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not read fingerprint file: {e}")
    
    index_path = os.path.join(indexes_folder, "index.faiss")
    metadata_path = os.path.join(indexes_folder, "metadata.json")
    print(f"[DEBUG] Current fingerprint: {fingerprint_hash}")
    print(f"[DEBUG] Previous fingerprint: {previous_fingerprint}")
    
    if (os.path.exists(index_path) and os.path.exists(metadata_path) and previous_fingerprint == fingerprint_hash):
        try:
            vectorstore = LC_FAISS.load_local(indexes_folder, embeddings)
            print("[INFO] Loaded existing vectorstore.")
            return vectorstore
        except Exception as e:
            print(f"[INFO] Failed to load existing vectorstore: {e}")
    
    print("[INFO] Building/rebuilding index from documents...")
    # Создаём чанки из документов
    chunks = process_document_folder(
        data_folder,
        min_words_per_page=100,
        target_chunk_size=512,
        min_chunk_size=256,
        overlap_size=150
    )
    if not chunks:
        vectorstore = LC_FAISS.from_documents([Document(page_content="Empty index", metadata={})], embeddings)
        vectorstore.save_local(indexes_folder)
        return vectorstore
    
    # Преобразуем чанки в объекты Document
    docs = [Document(page_content=ch["text"], metadata=ch["metadata"]) for ch in chunks]
    vectorstore = LC_FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(indexes_folder)
    
    with open(fingerprint_file, 'w') as f:
        json.dump(fingerprint_hash, f)
    
    return vectorstore

# ============================================================
# Шаблоны (prompt) для цепочек QA (отдельные для учителей и студентов)
# ============================================================
def get_teacher_prompt_template():
    return (
        "Ты — умный университетский ассистент для преподавателей и для сотрудников Университета AITU. Тебя зовут AITU - Connect. "
        "Используй следующий полученный контекст для ответа на вопрос. "
        "Сначала разберись с заданным вопросом, попытайся его полностью понять, разберись с контекстом. Подумай перед тем как ответить"
        "Внимание: в конце  твоего ответа ОБЯЗАТЕЛЬНО добавь отдельный раздел с заголовком 'Источники:', "
        "где перечисляй имена файлов с которых взял информацию и, если возможно, номера страниц. "
        "Отвечай емко, четко и полно, в несколько абзацев, разбивая ответ на логически структурированные абзацы, "
        "а при необходимости используй нумерованные списки или bullet points для наглядности. "
        "Будь дружелюбным, действуй как надежный друг и ассистент, старайся найти ответ на любой вопрос. Даже очень сложный. "
        "Если в базе данных недостаточно информации, сообщяй, что можно обратиться в офис регистратора.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )

def get_student_prompt_template():
    return (
        "Ты — умный университетский ассистент для студентов Университета AITU. "
        "Тебя зовут AITU - Connect. "
        "Используй следующий предоставленный контекст для ответа на вопрос. "
        "Сначала внимательно изучи вопрос, разберись в его сути и контексте. Подумай перед тем как ответить"
        "Отвечай емко, четко и полно, в несколько абзацев, разбивая ответ на логически структурированные абзацы, "
        "а при необходимости используй нумерованные списки или bullet points для наглядности. "
        "Будь дружелюбным, отзывчивым и всегда старайся помочь студентам с их университетскими запросами. "
        "Если в базе данных недостаточно информации, сообщи, что можно обратиться в офис поддержки.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )

# ============================================================
# Инициализация LLM и цепочек QA для учителей и студентов
# ============================================================
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
teacher_prompt = PromptTemplate(
    template=get_teacher_prompt_template(),
    input_variables=["context", "question"]
)
student_prompt = PromptTemplate(
    template=get_student_prompt_template(),
    input_variables=["context", "question"]
)

teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS)
student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS)

teacher_qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=teacher_vectorstore.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": teacher_prompt}
)

student_qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=student_vectorstore.as_retriever(),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": student_prompt}
)

# ============================================================
# Класс для управления историей чата с добавлением источников в ответ
# ============================================================
class ChatAssistant:
    def __init__(self, qa_chain):
        self.qa = qa_chain
        self.histories = {}  # {session_id: list of (role, content)}
    def get_answer(self, user_query: str, session_id: str = "default"):
        if session_id not in self.histories:
            self.histories[session_id] = []
        chain_history = self._convert_history(session_id)
        result = self.qa({"question": user_query, "chat_history": chain_history})
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        # Добавляем в конец ответа секцию "Sources:" с информацией об источниках
        if source_docs:
            sources = self._extract_sources(source_docs)
            if sources:
                sources_text = "\n\nSources:\n" + "\n".join(sources)
                answer += sources_text
        self.histories[session_id].append(("user", user_query))
        self.histories[session_id].append(("assistant", answer))
        return answer, source_docs
    def _convert_history(self, session_id: str):
        history = self.histories.get(session_id, [])
        pairs = []
        user_text = None
        for role, content in history:
            if role == "user":
                user_text = content
            elif role == "assistant":
                if user_text is not None:
                    pairs.append((user_text, content))
                    user_text = None
        return pairs
    def _extract_sources(self, source_docs) -> List[str]:
        sources = {}
        for doc in source_docs:
            metadata = doc.metadata
            if "file_name" not in metadata:
                continue
            file_name = metadata.get("file_name")
            if metadata.get("file_type") == "pdf" and "page_number" in metadata:
                if file_name not in sources:
                    sources[file_name] = []
                if metadata["page_number"] not in sources[file_name]:
                    sources[file_name].append(metadata["page_number"])
            else:
                if file_name not in sources:
                    sources[file_name] = []
        formatted_sources = []
        for file_name, pages in sources.items():
            if pages:
                pages.sort()
                formatted_sources.append(f"- {file_name} (Page {', '.join(str(p) for p in pages)})")
            else:
                formatted_sources.append(f"- {file_name}")
        return formatted_sources
    def clear_history(self, session_id: str = "default"):
        self.histories[session_id] = []

# Создаём объекты ChatAssistant для каждой модели
teacher_assistant = ChatAssistant(teacher_qa_chain)
student_assistant = ChatAssistant(student_qa_chain)

# ============================================================
# Утилита для извлечения списка источников (если нужно отдельно)
# ============================================================
def extract_sources_list(source_docs) -> List[str]:
    sources = {}
    for doc in source_docs:
        metadata = doc.metadata
        if "file_name" not in metadata:
            continue
        file_name = metadata.get("file_name")
        if metadata.get("file_type") == "pdf" and "page_number" in metadata:
            if file_name not in sources:
                sources[file_name] = []
            if metadata["page_number"] not in sources[file_name]:
                sources[file_name].append(metadata["page_number"])
        else:
            if file_name not in sources:
                sources[file_name] = []
    formatted_sources = []
    for file_name, pages in sources.items():
        if pages:
            pages.sort()
            formatted_sources.append(f"{file_name} (pages: {', '.join(str(p) for p in pages)})")
        else:
            formatted_sources.append(file_name)
    return formatted_sources

# ============================================================
# Инициализация менеджеров документов для учителей и студентов
# ============================================================
teacher_doc_manager = DocumentManager(DATA_FOLDER_TEACHERS)
student_doc_manager = DocumentManager(DATA_FOLDER_STUDENTS)

# ============================================================
# Модели данных для FastAPI
# ============================================================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

# ============================================================
# Создание приложения FastAPI и эндпойнтов
# ============================================================
app = FastAPI(title="University Chat Assistant API")

# Эндпойнт для вывода всех маршрутов приложения
@app.get("/api/endpoints")
def list_endpoints():
    routes = []
    for route in app.routes:
        if hasattr(route, "methods"):
            methods = ", ".join(route.methods)
            routes.append({
                "path": route.path,
                "methods": methods,
                "name": route.name
            })
    return routes

# ---- Чат-эндпойнты для учителей и студентов ----
@app.post("/api/teacher/chat", response_model=ChatResponse)
def teacher_chat(payload: ChatRequest):
    answer, source_docs = teacher_assistant.get_answer(payload.query, payload.session_id)
    sources_list = extract_sources_list(source_docs)
    return ChatResponse(answer=answer, sources=sources_list)

@app.post("/api/student/chat", response_model=ChatResponse)
def student_chat(payload: ChatRequest):
    answer, source_docs = student_assistant.get_answer(payload.query, payload.session_id)
    sources_list = extract_sources_list(source_docs)
    return ChatResponse(answer=answer, sources=sources_list)

# ---- Эндпойнты для работы с документами (список и загрузка) ----
@app.get("/api/teacher/docs")
def list_teacher_docs():
    return teacher_doc_manager.get_active_documents()

@app.get("/api/student/docs")
def list_student_docs():
    return student_doc_manager.get_active_documents()

@app.post("/api/teacher/docs/upload")
def upload_teacher_doc(
    file: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    tags: str = Form("")
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    with open(temp_filepath, "wb") as f:
        f.write(file.file.read())
    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    doc_info = teacher_doc_manager.add_document(
        temp_filepath,
        title=title,
        description=description,
        tags=tags_list
    )
    global teacher_vectorstore
    teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS)
    return {
        "message": "Документ успешно загружен",
        "doc_id": doc_info["id"],
        "title": doc_info["title"],
        "version": doc_info["version"]
    }

@app.post("/api/student/docs/upload")
def upload_student_doc(
    file: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    tags: str = Form("")
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    with open(temp_filepath, "wb") as f:
        f.write(file.file.read())
    tags_list = [t.strip() for t in tags.split(",") if t.strip()]
    doc_info = student_doc_manager.add_document(
        temp_filepath,
        title=title,
        description=description,
        tags=tags_list
    )
    global student_vectorstore
    student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS)
    return {
        "message": "Документ успешно загружен",
        "doc_id": doc_info["id"],
        "title": doc_info["title"],
        "version": doc_info["version"]
    }

# Дополнительный эндпойнт для очистки истории чата
@app.post("/api/{role}/chat/clear")
def clear_chat(role: str, session_id: str = "default"):
    if role.lower() == "teacher":
        teacher_assistant.clear_history(session_id)
    elif role.lower() == "student":
        student_assistant.clear_history(session_id)
    return {"message": "История чата очищена"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
