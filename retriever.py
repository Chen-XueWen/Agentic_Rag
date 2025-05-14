from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain.docstore.document import Document

class GuestInfoRetriever:
    def __init__(self, docs: List[Document]):
        self._bm25 = BM25Retriever.from_documents(docs)

    def query(self, q: str, top_k: int = 3) -> str:
        results = self._bm25.invoke(q)
        if not results:
            return "No matching guest information found."
        # join up to top_k matches
        return "\n\n".join(doc.page_content for doc in results[:top_k])
