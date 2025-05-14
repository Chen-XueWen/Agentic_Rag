import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_together import ChatTogether
from langfuse.callback import CallbackHandler

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

TOGETHER_API_KEY: str = "###"
LANGFUSE_PUBLIC_KEY: str = "###"
LANGFUSE_SECRET_KEY: str = "###"
LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

# Generate the chat interface, including the tools
chat = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", api_key=TOGETHER_API_KEY)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY,
                                   secret_key=LANGFUSE_SECRET_KEY,
                                   host=LANGFUSE_HOST)

#### Example 1 ####
response = alfred.invoke(input={"messages": "Tell me about 'Lady Ada Lovelace'"},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

#### Example 2 ####
response = alfred.invoke(input={"messages": "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)


#### Example 3 ####
response = alfred.invoke(input={"messages": "One of our guests is from Qwen. What can you tell me about their most popular model?"},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

#### Example 4 ####
response = alfred.invoke(input={"messages":"I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# Example 5: Conversation Memory
# First interaction
response = alfred.invoke(input={"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print()

# Second interaction (referencing the first)
response = alfred.invoke(input={"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]},
                         config={"callbacks": [langfuse_handler]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)