from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from app.config import DATA_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS
from data_management.document_manager import DocumentManager
from app.utils import extract_text_from_file, find_similar_files
from app.prompts import get_teacher_prompt_template, get_student_prompt_template
from app.vectorstore_singleton import get_teacher_vectorstore, get_student_vectorstore, get_llm, refresh_teacher_vectorstore, refresh_student_vectorstore
from langchain.prompts.prompt import PromptTemplate

router = APIRouter()

# Initialize document managers
teacher_doc_manager = DocumentManager(DATA_FOLDER_TEACHERS)
student_doc_manager = DocumentManager(DATA_FOLDER_STUDENTS)

# Get vectorstores and LLM from singleton
teacher_vectorstore = get_teacher_vectorstore()
student_vectorstore = get_student_vectorstore()
llm = get_llm()

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
    
    # Refresh the appropriate vectorstore
    if role == "teacher":
        refresh_teacher_vectorstore()
    else:
        refresh_student_vectorstore()
        
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