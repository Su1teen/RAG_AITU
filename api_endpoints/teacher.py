from fastapi import APIRouter
from pydantic import BaseModel
from app.chat_assistant import ChatAssistant
from app.utils import extract_sources_list
from app.config import DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, OPENAI_API_KEY
from data_management.vectorstore_utils import load_or_rebuild_vectorstore
from app.embeddings import embeddings
from app.prompts import get_teacher_prompt_template
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

router = APIRouter()

teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")
teacher_prompt = PromptTemplate(
    template=get_teacher_prompt_template(),
    input_variables=["chat_history", "context", "question"]
)
teacher_qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=teacher_vectorstore.as_retriever(search_kwargs={"k": 7}),
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": teacher_prompt}
)
teacher_assistant = ChatAssistant(teacher_qa_chain)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@router.post("/chat", response_model=ChatResponse)
def teacher_chat(payload: ChatRequest):
    answer, source_docs = teacher_assistant.get_answer(payload.query, payload.session_id)
    sources_list = extract_sources_list(source_docs)
    return ChatResponse(answer=answer, sources=sources_list)
