from fastapi import APIRouter
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from app.vectorstore_utils import load_or_rebuild_vectorstore
from app.history import ChatAssistant
from app.config import *
from langchain.chains import ConversationalRetrievalChain
from app.prompts import (
    get_student_flowchart_prompt,
    get_teacher_flowchart_prompt
)
student_flowchart_template = PromptTemplate(
    template=get_student_flowchart_prompt(),
    input_variables=['context','question']
)
student_flowchart_template = PromptTemplate(
    template=get_teacher_flowchart_prompt(),
    input_variables=['context','question']
)
router = APIRouter(prefix='/api')

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-4o-mini')
# аналогично инициализируем vs и assistant для teacher/student

doc = BaseModel  # заглушка

@router.post('/teacher/flowchart')
def teacher_flowchart(payload: dict):
    # содержимое соответствует основному main.py
    pass  # см. оригинал

@router.post('/student/flowchart')
def student_flowchart(payload: dict):
    pass  # аналогично