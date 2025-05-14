from langchain.tools import Tool
from retriever import GuestInfoRetriever

def build_guest_info_tool(retriever: GuestInfoRetriever) -> Tool:
    return Tool(
        name="guest_info_retriever",
        func=retriever.query,
        description="Retrieves detailed information about gala guests based on their name or relation."
    )