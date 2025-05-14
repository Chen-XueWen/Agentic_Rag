import os
from config import settings
from graph_builder import build_graph
from langchain_core.messages import HumanMessage
from langfuse.callback import CallbackHandler

def main():
    # ensure API keys are in the environment
    for key in (
        "TOGETHER_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ):
        assert os.getenv(key), f"{key} is not set"

    alfred = build_graph()
    # sample interaction
    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]

    handler = CallbackHandler()
    response = alfred.invoke(input={"messages": messages}, config={"callbacks": [handler]})

    print("🎩 Alfred's Response:")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    main()