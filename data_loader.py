from datasets import load_dataset
from langchain.docstore.document import Document
from config import settings

def load_guest_documents() -> list[Document]:
    ds = load_dataset(settings.DATASET_NAME, split=settings.DATASET_SPLIT)
    docs = []
    for g in ds:
        content = "\n".join([
            f"Name: {g['name']}",
            f"Relation: {g['relation']}",
            f"Description: {g['description']}",
            f"Email: {g['email']}",
        ])
        docs.append(Document(page_content=content, metadata={"name": g["name"]}))
    return docs