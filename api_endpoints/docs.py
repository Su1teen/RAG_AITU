from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from app.config import DATA_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS, INDEXES_FOLDER_TEACHERS, INDEXES_FOLDER_STUDENTS, OPENAI_API_KEY
from data_management.document_manager import DocumentManager
from data_management.vectorstore_utils import load_or_rebuild_vectorstore
from app.utils import extract_text_from_file, find_similar_files
from app.prompts import get_teacher_prompt_template, get_student_prompt_template
from app.embeddings import embeddings
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

router = APIRouter()

teacher_doc_manager = DocumentManager(DATA_FOLDER_TEACHERS)
student_doc_manager = DocumentManager(DATA_FOLDER_STUDENTS)

teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS)
student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

@router.get("/teacher/docs")
def list_teacher_docs():
    return teacher_doc_manager.get_active_documents()

@router.get("/student/docs")
def list_student_docs():
    return student_doc_manager.get_active_documents()

@router.post("/{role}/docs/upload")
async def upload_doc(
    role: str,
    file: UploadFile = File(...),
    replace_doc_id: str = Form(None)
):
    data_folder = DATA_FOLDER_TEACHERS if role == "teacher" else DATA_FOLDER_STUDENTS
    mgr = teacher_doc_manager if role == "teacher" else student_doc_manager
    content = await file.read()
    temp_path = os.path.join(data_folder, file.filename)
    with open(temp_path, "wb") as f:
        f.write(content)
    if replace_doc_id:
        mgr.delete_document_by_id(replace_doc_id)
        new_doc = mgr.add_document(temp_path)
        action = f"Replaced {replace_doc_id} → {new_doc['id']}"
    else:
        new_doc = mgr.add_document(temp_path)
        action = f"Added new document {new_doc['id']}"
    # DO NOT rebuild index here! Only update metadata.
    return {"message": "Done", "file_action": action}

@router.post("/{role}/docs/check_similarity")
async def check_doc_similarity(
    role: str,
    file: UploadFile = File(...)
):
    folder = DATA_FOLDER_TEACHERS if role == "teacher" else DATA_FOLDER_STUDENTS
    tmp = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(await file.read())
    text = extract_text_from_file(tmp)
    dups = find_similar_files(text, folder, threshold=0.7)
    os.remove(tmp)
    return {"possible_duplicates": dups}

# --- Analyze endpoints for instant file analysis (teacher: web, student: telegram) ---

teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS)
student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

@router.post("/teacher/docs/analyze")
async def analyze_teacher_doc(file: UploadFile = File(...), question: str = Form("")):
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        print(f"[DEBUG] Temp file path: {tmp_path}")
        file_text = extract_text_from_file(tmp_path)
        print(f"[DEBUG] Extracted text length: {len(file_text)}")
        os.remove(tmp_path)
        if not file_text.strip():
            return {"answer": "Файл не содержит данных или его формат не поддерживается. Пожалуйста, загрузите .docx, .pdf, .txt, .xlsx, .xls или .pptx файл с текстом."}
        prompt_text = question if question is not None else ""
        relevant_docs = teacher_vectorstore.similarity_search(prompt_text, k=3)
        context = "\n".join(doc.page_content for doc in relevant_docs)
        prompt = PromptTemplate(
            template=get_teacher_prompt_template(),
            input_variables=["chat_history", "context", "question"]
        )
        chain = prompt | llm
        chain_response = chain.invoke({
            "chat_history": file_text,
            "context": context,
            "question": prompt_text
        })
        return {"answer": chain_response.content.strip()}
    except Exception as e:
        print(f"[ERROR] Exception in analyze_teacher_doc: {e}")
        return {"error": str(e)}

@router.post("/student/docs/analyze")
async def analyze_student_doc(file: UploadFile = File(...), question: str = Form("")):
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        print(f"[DEBUG] Temp file path: {tmp_path}")
        file_text = extract_text_from_file(tmp_path)
        print(f"[DEBUG] Extracted text length: {len(file_text)}")
        os.remove(tmp_path)
        if not file_text.strip():
            return {"answer": "Файл не содержит данных или его формат не поддерживается. Пожалуйста, загрузите .docx, .pdf, .txt, .xlsx, .xls или .pptx файл с текстом."}
        # If prompt is empty, just analyze the file text
        prompt_text = question if question is not None else ""
        relevant_docs = student_vectorstore.similarity_search(prompt_text, k=3)
        context = "\n".join(doc.page_content for doc in relevant_docs)
        prompt = PromptTemplate(
            template=get_student_prompt_template(),
            input_variables=["chat_history", "context", "question"]
        )
        chain = prompt | llm
        chain_response = chain.invoke({
            "chat_history": file_text,
            "context": context,
            "question": prompt_text
        })
        return {"answer": chain_response.content.strip()}
    except Exception as e:
        print(f"[ERROR] Exception in analyze_student_doc: {e}")
        return {"error": str(e)}
