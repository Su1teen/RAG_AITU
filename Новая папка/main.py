from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import chat, docs, flowchart# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# импорт промптов
from app.prompts import (
    get_student_prompt_template,
    get_teacher_prompt_template,
    get_student_flowchart_prompt,
    get_teacher_flowchart_prompt,
)

# импорт LangChain компонентов
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ваши векторные стораджи, эмбеддинги и т.д.
from app.config import (
    OPENAI_API_KEY,
    DATA_FOLDER_STUDENTS,
    INDEXES_FOLDER_STUDENTS,
    DATA_FOLDER_TEACHERS,
    INDEXES_FOLDER_TEACHERS,
)

from app.embeddings import MyEmbeddings
from app.vectorstore_utils import load_or_rebuild_vectorstore

# ——————————————————————————————————————————————
# 1. Инициализируем LLM и PromptTemplate
# ——————————————————————————————————————————————
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

# Тут создаём сами объекты PromptTemplate на основе ваших функций
student_prompt = PromptTemplate(
    template=get_student_prompt_template(),
    input_variables=["chat_history", "context", "question"]
)
teacher_prompt = PromptTemplate(
    template=get_teacher_prompt_template(),
    input_variables=["chat_history", "context", "question"]
)

# Аналогично для flowchart-шаблонов, если нужно:
student_flowchart_prompt = PromptTemplate(
    template=get_student_flowchart_prompt(),
    input_variables=["context", "question"]
)
teacher_flowchart_prompt = PromptTemplate(
    template=get_teacher_flowchart_prompt(),
    input_variables=["context", "question"]
)

# ——————————————————————————————————————————————
# 2. Загружаем/перестраиваем FAISS и строим цепочки
# ——————————————————————————————————————————————

# Студентский сторадж
emb = MyEmbeddings()
vs_student = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS, emb)
student_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vs_student.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": student_prompt}
)

# Преподавательский сторадж
vs_teacher = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, emb)
teacher_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vs_teacher.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": teacher_prompt}
)

# ——————————————————————————————————————————————
# 3. Подключаем роутеры и запускаем приложение
# ——————————————————————————————————————————————

from app.routers import chat, docs, flowchart

app = FastAPI(title="University Chat Assistant API")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.include_router(chat.router)
app.include_router(docs.router)
app.include_router(flowchart.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


app = FastAPI(title='University Chat Assistant API')
app.mount('/static', StaticFiles(directory='static', html=True), name='static')
app.include_router(chat.router)
app.include_router(docs.router)
app.include_router(flowchart.router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)