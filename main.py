from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api_endpoints.teacher import router as teacher_router
from api_endpoints.student import router as student_router
from api_endpoints.flowchart import router as flowchart_router
from api_endpoints.docs import router as docs_router
from api_endpoints.chat import router as chat_router
from api_endpoints.generate import router as generate_router
#from api_endpoints.syllabus import router as syllabus_router

from app.config import DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS
from data_management.vectorstore_utils import load_or_rebuild_vectorstore
from api_endpoints import teacher, student, chat  # Import your router modules

# Import the global variables that need to be modified
from api_endpoints.teacher import teacher_vectorstore, teacher_qa_chain
from api_endpoints.student import student_vectorstore, student_qa_chain
from fastapi import HTTPException

app = FastAPI(title="University Chat Assistant API")

# app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.include_router(teacher_router, prefix="/api/teacher")
app.include_router(student_router, prefix="/api/student")
app.include_router(flowchart_router, prefix="/api")
app.include_router(docs_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(generate_router, prefix="/api")
app.include_router(syllabus_router)  # Remove prefix="/api" since router already has it
@app.post("/refresh/staff")
async def refresh_staff_index():
    try:
        global teacher_vectorstore, teacher_qa_chain
        print(f"[DEBUG] Starting refresh for staff index")
        teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS,
                                                          call_id="refresh_staff")

        # Update the QA chain with the new retriever
        teacher_qa_chain.retriever = teacher_vectorstore.as_retriever(search_kwargs={"k": 3})

        print(f"[DEBUG] Rebuilt teacher_vectorstore with {teacher_vectorstore.index.ntotal} vectors")
        return {"message": "Индекс для сотрудников (Teacher) был успешно пересобран"}
    except Exception as e:
        print(f"[ERROR] Failed to refresh staff index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh/students")
def refresh_students_index():
    """
    Принудительно пересобрать индекс для студентов.
    """
    global student_vectorstore, student_qa_chain
    student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS)

    # Update the QA chain with the new retriever
    student_qa_chain.retriever = student_vectorstore.as_retriever(search_kwargs={"k": 3})

    return {"message": "Индекс для студентов был успешно пересобран"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)