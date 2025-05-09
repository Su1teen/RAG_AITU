from datetime import datetime
from typing import List

class ChatAssistant:
    def __init__(self, qa_chain):
        self.qa = qa_chain
        self.histories = {}  # {session_id: list of dicts}

    def get_answer(self, user_query: str, session_id: str = "default"):
        if session_id not in self.histories:
            self.histories[session_id] = []
        chain_history = self._convert_history(session_id)
        result = self.qa({
            "question": user_query,
            "chat_history": chain_history
        })
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        sources = self._extract_sources(source_docs)
        self.histories[session_id].append({
            "role": "user",
            "content": user_query,
            "time": datetime.now().strftime("%I:%M %p"),
        })
        self.histories[session_id].append({
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%I:%M %p"),
            "sources": sources
        })
        return answer, source_docs

    def _convert_history(self, session_id: str):
        history = self.histories.get(session_id, [])
        pairs = []
        last_user = None
        for entry in history:
            role = entry.get("role")
            content = entry.get("content")
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, content))
                last_user = None
        return pairs

    def _extract_sources(self, source_docs) -> List[str]:
        seen = set()
        sources = []
        for doc in source_docs:
            file_name = doc.metadata.get("file_name")
            if file_name and file_name not in seen:
                seen.add(file_name)
                sources.append(file_name)
        return sources

    def clear_history(self, session_id: str = "default"):
        self.histories[session_id] = []
