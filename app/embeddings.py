from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

class MyEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()