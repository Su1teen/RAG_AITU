from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.chat_assistant import ChatAssistant
from app.config import DATA_FOLDER_TEACHERS, DATA_FOLDER_STUDENTS
from data_management.document_manager import DocumentManager
from typing import List, Dict, Any, Optional

router = APIRouter()

teacher_doc_manager = DocumentManager(DATA_FOLDER_TEACHERS)
student_doc_manager = DocumentManager(DATA_FOLDER_STUDENTS)

# Dummy ChatAssistant objects for import; replace with actual shared instances if needed
teacher_assistant = ChatAssistant(None)
student_assistant = ChatAssistant(None)

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]

class ChatDeleteResponse(BaseModel):
    message: str

@router.get("/{role}/chat/clear")
def clear_chat(role: str, session_id: str = "default"):
    if role.lower() == "teacher":
        teacher_assistant.clear_history(session_id)
    elif role.lower() == "student":
        student_assistant.clear_history(session_id)
    else:
        raise HTTPException(status_code=404, detail="Role not found")
    return {"message": "История чата очищена"}

@router.get("/{role}/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(role: str, session_id: str = "default"):
    if role.lower() == "teacher":
        hist = teacher_assistant.histories.get(session_id, [])
    elif role.lower() == "student":
        hist = student_assistant.histories.get(session_id, [])
    else:
        raise HTTPException(status_code=404, detail="Role not found")
    conversation = [
        {
            "id": idx,
            "role": entry["role"],
            "content": entry["content"],
            "time": entry.get("time"),
            "sources": entry.get("sources", [])
        }
        for idx, entry in enumerate(hist)
    ]
    return {"session_id": session_id, "history": conversation}

@router.delete("/{role}/chat/history", response_model=ChatDeleteResponse)
def delete_chat_message(
    role: str,
    session_id: str = "default",
    message_id: int = None
):
    if role.lower() == "teacher":
        hist = teacher_assistant.histories.get(session_id, [])
    elif role.lower() == "student":
        hist = student_assistant.histories.get(session_id, [])
    else:
        raise HTTPException(status_code=404, detail="Role not found")
    if message_id is None or message_id < 0 or message_id >= len(hist):
        raise HTTPException(status_code=400, detail="Invalid message_id")
    hist.pop(message_id)
    return {"message": f"Deleted message {message_id} from session {session_id}"}
