"""
LangGraph React Tutorial - Converted from Jupyter Notebook
This script demonstrates a simple ReAct agent using LangGraph with a weather tool.
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define the weather tool
@tool
# def get_weather(location: str):
def xyz(location: str):
    """Call to get the current weather."""
    if location.lower() in ["yorkshire"]:
        return "It's cold and wet."
    else:
        return "It's warm and sunny."


# Initialize the LLM
# IMPORTANT: Use environment variable for API key instead of hardcoding
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Set this in your environment
    model="gpt-4o",
)

# Bind tools to LLM
tools = [xyz]
llm_with_tools = llm.bind_tools(tools)

# Create the graph
graph = StateGraph(State)

# Create tool node
tool_node = ToolNode(tools)
graph.add_node("tool_node", tool_node)


# Define the prompt node
def prompt_node(state: State) -> State:
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}


graph.add_node("prompt_node", prompt_node)


# Define conditional edge logic
def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"


# Add edges to the graph
graph.add_conditional_edges(
    'prompt_node',
    conditional_edge
)
graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")

# Compile the graph
APP = graph.compile()


# Main execution
if __name__ == "__main__":
    # Run the agent
    new_state = APP.invoke({"messages": ["Whats the weather in yorkshire?"]})
    
    # Print the response
    print(new_state["messages"][-1].content)