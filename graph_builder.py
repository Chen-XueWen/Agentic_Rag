from langchain_core.messages import AnyMessage
from langchain.docstore.document import Document
from langchain_together import ChatTogether
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, StateGraph
from typing import TypedDict, Annotated
from config import settings
from tools import build_guest_info_tool
from retriever import GuestInfoRetriever
from data_loader import load_guest_documents

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def build_graph():
    # load docs & retriever
    docs = load_guest_documents()
    retriever = GuestInfoRetriever(docs)

    # build tool
    guest_tool = build_guest_info_tool(retriever)
    tools = [guest_tool]

    # init LLM
    # chat = ChatOpenAI(model="gpt-4o") # Use openAI if there is image
    chat = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    chat_with_tools = chat.bind_tools(tools)

    # assistant node
    def assistant(state: AgentState):
        return {"messages": [chat_with_tools.invoke(state["messages"])]}

    # build state graph
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
