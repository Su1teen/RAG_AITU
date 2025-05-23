from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils import extract_sources_list
from langchain.prompts.prompt import PromptTemplate
from app.config import DATA_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS, OPENAI_API_KEY, INDEXES_FOLDER_TEACHERS, INDEXES_FOLDER_STUDENTS
from data_management.vectorstore_utils import load_or_rebuild_vectorstore
from app.prompts import get_teacher_flowchart_prompt, get_student_flowchart_prompt
from app.embeddings import embeddings
from app.vectorstore_singleton import get_teacher_vectorstore, get_student_vectorstore, get_llm
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS as LC_FAISS

router = APIRouter()
teacher_vectorstore = get_teacher_vectorstore()
student_vectorstore = get_student_vectorstore()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

@router.post("/teacher/flowchart")
def teacher_flowchart(payload: ChatRequest):
    relevant_docs = teacher_vectorstore.similarity_search(payload.query, k=3)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    sources = extract_sources_list(relevant_docs)
    prompt = PromptTemplate(
        template=get_teacher_flowchart_prompt(),
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    chain_response = chain.invoke({"context": context, "question": payload.query})
    mermaid_code = chain_response.content.strip()
    return JSONResponse({
        "mermaid": mermaid_code,
        "sources": sources
    })

@router.post("/student/flowchart")
def student_flowchart(payload: ChatRequest):
    relevant_docs = student_vectorstore.similarity_search(payload.query, k=3)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    sources = extract_sources_list(relevant_docs)
    prompt = PromptTemplate(
        template=get_student_flowchart_prompt(),
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    chain_response = chain.invoke({"context": context, "question": payload.query})
    mermaid_code = chain_response.content.strip()
    return JSONResponse({
        "mermaid": mermaid_code,
        "sources": sources
    })
