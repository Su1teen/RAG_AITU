from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel

class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict]

class ChatDeleteResponse(BaseModel):
    message: str

class ChatAssistant:
    def __init__(self, qa_chain):
        self.qa = qa_chain
        self.histories = {}

    def _extract_sources(self, docs) -> List[str]:
        seen, out = set(), []
        for d in docs:
            fn = d.metadata.get('file_name')
            if fn and fn not in seen:
                seen.add(fn); out.append(fn)
        return out

    def _to_pairs(self, session_id: str):
        hist = self.histories.get(session_id, [])
        pairs, last_user = [], None
        for e in hist:
            if e['role']=='user': last_user=e['content']
            elif last_user:
                pairs.append((last_user, e['content'])); last_user=None
        return pairs

    def get_answer(self, query: str, session_id: str='default'):
        if session_id not in self.histories:
            self.histories[session_id] = []
        history_pairs = self._to_pairs(session_id)
        res = self.qa({'question': query, 'chat_history': history_pairs})
        ans = res.get('answer','')
        docs = res.get('source_documents', [])
        sources = self._extract_sources(docs)
        ts = datetime.now().strftime('%H:%M')
        self.histories[session_id] += [
            {'role':'user','content':query,'time':ts},
            {'role':'assistant','content':ans,'time':ts,'sources':sources}
        ]
        return ans, sources

    def clear_history(self, session_id: str='default'):
        self.histories[session_id] = []