import os
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.config import *
from app.embeddings import MyEmbeddings
from app.vectorstore_utils import load_or_rebuild_vectorstore
from app.history import ChatAssistant, ChatHistoryResponse, ChatDeleteResponse
from app.file_utils import extract_text_from_file, find_similar_files
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from app.prompts import get_student_prompt_template
router = APIRouter(prefix='/api')

# инициализация ресурсов
emb = MyEmbeddings()
vs_student = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS, emb)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name='gpt-4o-mini')
# шаблон prompt для student
student_prompt = PromptTemplate(
    template=get_student_prompt_template(),
    input_variables=['chat_history','context','question']
)
student_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vs_student.as_retriever(search_kwargs={'k':5}),
    return_source_documents=True,
    combine_docs_chain_kwargs={'prompt': student_prompt}
)
student_assistant = ChatAssistant(student_chain)

class QueryForm(BaseModel):
    query: str
    session_id: Optional[str] = 'default'

@router.post('/student/chat', response_model=ChatHistoryResponse)
def student_chat(
    query: str = Form(...), session_id: str = Form('default')
):
    ans, _ = student_assistant.get_answer(query, session_id)
    return {'session_id': session_id, 'history': student_assistant.histories[session_id]}

@router.post('/student/chat_with_file', response_model=ChatHistoryResponse)
async def student_chat_file(
    question: str = Form(...), file: UploadFile = File(...)
):
    tmp = os.path.join(TMP_DIR, file.filename)
    with open(tmp,'wb') as f: f.write(await file.read())
    text = extract_text_from_file(tmp)
    # разбивка на чанки и вставка в индекс
    from document_processor import chunk_text
    chunks = chunk_text(text)
    embeds = emb.embed_documents([c.text for c in chunks])
    temp_ids = vs_student.add(chunks, embeds)
    # QA
    retr = vs_student.as_retriever(k=3)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retr,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': student_prompt}
    )
    res = chain({'question': question, 'chat_history': []})
    ans = res['answer']
    sources = [d.metadata.get('file_name') for d in res['source_documents']]
    vs_student.delete(temp_ids)
    os.remove(tmp)
    student_assistant.clear_history('default')
    student_assistant.histories['default'].append({'role':'assistant','content': ans,'time':'','sources':sources})
    return {'session_id': 'default', 'history': student_assistant.histories['default']}

@router.get('/student/chat/clear')
def clear_student(session_id: str='default'):
    student_assistant.clear_history(session_id)
    return {'message': 'История чата очищена'}

@router.get('/student/chat/history', response_model=ChatHistoryResponse)
def history_student(session_id: str='default'):
    return {'session_id': session_id, 'history': student_assistant.histories.get(session_id, [])}

@router.post('/student/docs/check_similarity')
async def student_check_similarity(file: UploadFile = File(...)):
    tmp = os.path.join(TMP_DIR, file.filename)
    with open(tmp,'wb') as f: f.write(await file.read())
    text = extract_text_from_file(tmp)
    sims = find_similar_files(text, DATA_FOLDER_STUDENTS)
    os.remove(tmp)
    return {'possible_duplicates': sims}