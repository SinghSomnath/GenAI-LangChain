"""
LangGraph Router Agent - AstraDB vs Wikipedia
This script demonstrates a routing agent that decides whether to use
AstraDB vector search or Wikipedia search based on the user's question.

Python Version: 3.13.1
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# Third-party imports for AstraDB and Wikipedia
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()


# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(TypedDict):
    """State structure for the routing agent"""
    messages: Annotated[list, add_messages]
    route_decision: str  # "astradb" or "wikipedia"
    retrieved_docs: list  # Documents retrieved from either source


# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize the LLM for routing and final response
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0,
)

# Initialize embeddings for AstraDB
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    # model="text-embedding-3-small"
     model="text-embedding-3-large",
     dimensions=1024
)

# Initialize AstraDB Vector Store
# Make sure to set these environment variables:
# ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_KEYSPACE
astra_vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=os.getenv("ASTRA_DB_KEYSPACE"),
)

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=4000
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)


# ============================================================================
# GRAPH NODES
# ============================================================================

def router_node(state: State) -> State:
    """
    Routes the question to either AstraDB or Wikipedia.
    
    Uses LLM to analyze the question and decide which data source
    is most appropriate.
    """
    user_question = state["messages"][-1].content
    
    # Create a routing prompt
    routing_prompt = f"""You are a routing assistant. Analyze the following question and decide whether it should be answered using:

1. **ASTRADB**: A vector database containing domain-specific documents (e.g., company docs, technical manuals, custom knowledge base)
2. **WIKIPEDIA**: General knowledge encyclopedia for broad topics, historical facts, famous people, etc.

Question: {user_question}

Respond with ONLY one word: either "ASTRADB" or "WIKIPEDIA"

Guidelines:
- Use ASTRADB for: domain-specific queries, internal company info, technical documentation, specialized knowledge
- Use WIKIPEDIA for: general knowledge, famous people, historical events, scientific concepts, geography, etc.
"""
    
    routing_messages = [
        SystemMessage(content="You are a routing decision maker. Respond with only 'ASTRADB' or 'WIKIPEDIA'."),
        HumanMessage(content=routing_prompt)
    ]
    
    response = llm.invoke(routing_messages)
    route_decision = response.content.strip().upper()
    
    # Validate the decision
    if route_decision not in ["ASTRADB", "WIKIPEDIA"]:
        # Default to Wikipedia if unclear
        route_decision = "WIKIPEDIA"
    
    print(f"🔀 Routing Decision: {route_decision}")
    
    return {
        "route_decision": route_decision,
        "messages": [AIMessage(content=f"Routing to {route_decision}...")]
    }


def astradb_retrieval_node(state: State) -> State:
    """
    Retrieves relevant documents from AstraDB vector store.
    """
    user_question = state["messages"][0].content  # Get original question
    
    print(f"📚 Searching AstraDB for: {user_question}")
    
    try:
        # Perform similarity search in AstraDB
        results = astra_vector_store.similarity_search(
            query=user_question,
            k=4  # Retrieve top 4 most relevant documents
        )
        
        # Format the retrieved documents
        retrieved_docs = []
        for idx, doc in enumerate(results, 1):
            doc_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "AstraDB",
                "rank": idx
            }
            retrieved_docs.append(doc_info)
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents from AstraDB")
        
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [AIMessage(content=f"Retrieved {len(retrieved_docs)} documents from AstraDB")]
        }
    
    except Exception as e:
        print(f"❌ Error retrieving from AstraDB: {e}")
        return {
            "retrieved_docs": [],
            "messages": [AIMessage(content=f"Error retrieving from AstraDB: {str(e)}")]
        }


def wikipedia_retrieval_node(state: State) -> State:
    """
    Retrieves relevant information from Wikipedia.
    """
    user_question = state["messages"][0].content  # Get original question
    
    print(f"🌐 Searching Wikipedia for: {user_question}")
    
    try:
        # Search Wikipedia
        wiki_results = wikipedia_tool.run(user_question)
        
        # Format the retrieved documents
        retrieved_docs = [{
            "content": wiki_results,
            "metadata": {"source": "Wikipedia"},
            "source": "Wikipedia",
            "rank": 1
        }]
        
        print(f"✅ Retrieved information from Wikipedia")
        
        return {
            "retrieved_docs": retrieved_docs,
            "messages": [AIMessage(content="Retrieved information from Wikipedia")]
        }
    
    except Exception as e:
        print(f"❌ Error retrieving from Wikipedia: {e}")
        return {
            "retrieved_docs": [],
            "messages": [AIMessage(content=f"Error retrieving from Wikipedia: {str(e)}")]
        }


def response_generator_node(state: State) -> State:
    """
    Generates final response using the retrieved documents.
    """
    user_question = state["messages"][0].content
    retrieved_docs = state["retrieved_docs"]
    
    if not retrieved_docs:
        final_response = "I apologize, but I couldn't retrieve any relevant information to answer your question."
        return {"messages": [AIMessage(content=final_response)]}
    
    # Prepare context from retrieved documents
    context_parts = []
    for doc in retrieved_docs:
        source_label = f"[{doc['source']} - Rank {doc['rank']}]"
        context_parts.append(f"{source_label}\n{doc['content']}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Create response generation prompt
    response_prompt = f"""Based on the following retrieved information, answer the user's question accurately and comprehensively.

User Question: {user_question}

Retrieved Information:
{context}

Instructions:
- Provide a clear, accurate answer based on the retrieved information
- Cite the source when relevant (AstraDB or Wikipedia)
- If the information doesn't fully answer the question, say so
- Be concise but complete

Answer:"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on retrieved information."),
        HumanMessage(content=response_prompt)
    ]
    
    response = llm.invoke(messages)
    final_response = response.content
    
    print(f"✨ Generated final response")
    
    return {"messages": [AIMessage(content=final_response)]}


# ============================================================================
# CONDITIONAL EDGE LOGIC
# ============================================================================

def route_to_retrieval(state: State) -> Literal['astradb_retrieval', 'wikipedia_retrieval']:
    """
    Determines which retrieval node to use based on routing decision.
    """
    route_decision = state.get("route_decision", "WIKIPEDIA")
    
    if route_decision == "ASTRADB":
        return "astradb_retrieval"
    else:
        return "wikipedia_retrieval"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_graph():
    """
    Constructs the LangGraph routing agent.
    
    Flow:
    1. router_node: Decides between AstraDB or Wikipedia
    2. Conditional edge routes to appropriate retrieval node
    3. Retrieval node fetches relevant documents
    4. response_generator_node creates final answer
    """
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("astradb_retrieval", astradb_retrieval_node)
    graph.add_node("wikipedia_retrieval", wikipedia_retrieval_node)
    graph.add_node("response_generator", response_generator_node)
    
    # Set entry point 
    graph.set_entry_point("router")
    
    # Add conditional edge from router to retrieval nodes
    graph.add_conditional_edges(
        "router",
        route_to_retrieval,
        {
            "astradb_retrieval": "astradb_retrieval",
            "wikipedia_retrieval": "wikipedia_retrieval"
        }
    )
    
    # Add edges from retrieval nodes to response generator
    graph.add_edge("astradb_retrieval", "response_generator")
    graph.add_edge("wikipedia_retrieval", "response_generator")
    
    # Response generator is the end
    graph.add_edge("response_generator", "__end__")
    
    return graph.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(question: str):
    """
    Run the routing agent with a given question.
    
    Args:
        question: User's question to answer
    
    Returns:
        Final state containing the response
    """
    print(f"\n{'='*70}")
    print(f"❓ Question: {question}")
    print(f"{'='*70}\n")
    
    APP = build_graph()
    
    # Invoke the agent
    final_state = APP.invoke({
        "messages": [HumanMessage(content=question)],
        "route_decision": "",
        "retrieved_docs": []
    })
    
    # Print the final response
    print(f"\n{'='*70}")
    print(f"💬 FINAL RESPONSE:")
    print(f"{'='*70}")
    print(final_state["messages"][-1].content)
    print(f"{'='*70}\n")
    
    return final_state


if __name__ == "__main__":
    # Example 1: General knowledge question (should route to Wikipedia)
    # run_agent("Who is Shah Rukh Khan?")
    
    # Example 2: Technical/domain-specific question (should route to AstraDB)
    # Uncomment and customize based on your AstraDB content
    run_agent("What are the security best practices in our company handbook?")
    
    # Example 3: Another Wikipedia example
    # run_agent("What is the history of the Eiffel Tower?")
    
    # Example 4: Interactive mode
    # print("\n" + "="*70)
    # print("Interactive Mode - Enter your questions (or 'quit' to exit)")
    # print("="*70 + "\n")
    
    # while True:
    #     user_input = input("Your question: ").strip()
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         print("Goodbye!")
    #         break
    #     if user_input:
    #         run_agent(user_input)
