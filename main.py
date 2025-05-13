from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api_endpoints.teacher import router as teacher_router
from api_endpoints.student import router as student_router
from api_endpoints.flowchart import router as flowchart_router
from api_endpoints.docs import router as docs_router
from api_endpoints.chat import router as chat_router
from api_endpoints.generate import router as generate_router
from api_endpoints.syllabus import router as syllabus_router

app = FastAPI(title="University Chat Assistant API")

# app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.include_router(teacher_router, prefix="/api/teacher")
app.include_router(student_router, prefix="/api/student")
app.include_router(flowchart_router, prefix="/api")
app.include_router(docs_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(generate_router, prefix="/api")
app.include_router(syllabus_router)  # Remove prefix="/api" since router already has it

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)