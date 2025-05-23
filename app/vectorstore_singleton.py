from data_management.vectorstore_utils import load_or_rebuild_vectorstore
from app.config import DATA_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS, INDEXES_FOLDER_TEACHERS, INDEXES_FOLDER_STUDENTS
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
from app.embeddings import embeddings

# Initialize vectorstores once
print("[INFO] Initializing vectorstores in singleton...")
teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, embeddings)
student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS, embeddings)

# Initialize LLM once
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4o-mini")

def get_teacher_vectorstore():
    return teacher_vectorstore

def get_student_vectorstore():
    return student_vectorstore

def get_llm():
    return llm

def refresh_teacher_vectorstore():
    global teacher_vectorstore
    print("[INFO] Refreshing teacher vectorstore...")
    teacher_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_TEACHERS, INDEXES_FOLDER_TEACHERS, embeddings)
    return teacher_vectorstore

def refresh_student_vectorstore():
    global student_vectorstore
    print("[INFO] Refreshing student vectorstore...")
    student_vectorstore = load_or_rebuild_vectorstore(DATA_FOLDER_STUDENTS, INDEXES_FOLDER_STUDENTS, embeddings)
    return student_vectorstore