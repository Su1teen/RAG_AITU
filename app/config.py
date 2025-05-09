import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_FOLDER_TEACHERS = os.getenv("DATA_FOLDER", "data")
INDEXES_FOLDER_TEACHERS = os.getenv("INDEXES_FOLDER", "indexes")
DATA_FOLDER_STUDENTS = os.getenv("DATA_FOLDER_STUD", "data_stud")
INDEXES_FOLDER_STUDENTS = os.getenv("INDEXES_FOLDER_STUD", "indexes_stud")
