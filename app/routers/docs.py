from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from document_manager import DocumentManager
from app.vectorstore_utils import load_or_rebuild_vectorstore
from app.config import *

router = APIRouter(prefix='/api')

teacher_mgr = DocumentManager(DATA_FOLDER_TEACHERS)
student_mgr = DocumentManager(DATA_FOLDER_STUDENTS)

@router.get('/teacher/docs')
def list_teacher_docs():
    return teacher_mgr.get_active_documents()

@router.get('/student/docs')
def list_student_docs():
    return student_mgr.get_active_documents()

@router.post('/{role}/docs/upload')
async def upload_doc(
    role: str,
    file: UploadFile = File(...),
    replace_doc_id: str = Form(None)
):
    mgr = teacher_mgr if role=='teacher' else student_mgr
    folder = DATA_FOLDER_TEACHERS if role=='teacher' else DATA_FOLDER_STUDENTS
    idx_folder = INDEXES_FOLDER_TEACHERS if role=='teacher' else INDEXES_FOLDER_STUDENTS
    # сохранение
    tmp_path = os.path.join(folder, file.filename)
    content = await file.read()
    with open(tmp_path,'wb') as f: f.write(content)
    if replace_doc_id:
        mgr.delete_document_by_id(replace_doc_id)
        new = mgr.add_document(tmp_path)
        action = f"Replaced {replace_doc_id} -> {new['id']}"
    else:
        new = mgr.add_document(tmp_path)
        action = f"Added {new['id']}"
    # rebuild index
    vs = load_or_rebuild_vectorstore(folder, idx_folder, emb)
    return {'message':'Done','file_action': action}

@router.post('/{role}/docs/check_similarity')
def check_sim(role: str, file: UploadFile = File(...)):
    # реализовано в chat.py для student
    raise HTTPException(status_code=404, detail='Use /student/docs/check_similarity')

@router.post('/refresh/staff')
def refresh_staff():
    load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, emb)
    return {'message':'Staff index refreshed'}

@router.post('/refresh/students')
def refresh_students():
    load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS, emb)
    return {'message':'Student index refreshed'}