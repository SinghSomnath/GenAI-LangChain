"""
LangGraph Router Agent - AstraDB vs Wikipedia

Routes user questions to either AstraDB vector search or Wikipedia search,
then generates a final answer from the retrieved context.

Required .env variables:
    OPENROUTER_API_KEY (or OPENAI_API_KEY)  – LLM & embedding credentials
    ASTRA_DB_API_ENDPOINT                   – AstraDB Data API endpoint
    ASTRA_DB_APPLICATION_TOKEN              – AstraDB application token
    ASTRA_DB_COLLECTION                     – (optional, default: "default_collection")
    ASTRA_DB_KEYSPACE                       – (optional, default: "default_keyspace")

Python Version: 3.13.1
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from config import create_embeddings, create_llm, create_vector_store


# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(TypedDict):
    """State structure for the routing agent."""
    messages: Annotated[list, add_messages]
    route_decision: str          # "ASTRADB" or "WIKIPEDIA"
    retrieved_docs: list         # Documents retrieved from either source


# ============================================================================
# CONFIGURATION
# ============================================================================

# LLM for routing and final response generation
llm = create_llm(
    model="qwen/qwen3-235b-a22b",
    temperature=0,
    app_title="LangGraph Router Agent",
)

# Embeddings & AstraDB vector store (with validation, retry, mismatch handling)
embeddings = create_embeddings()
astra_vector_store = create_vector_store(embeddings)

# Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=4000,
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)


# ============================================================================
# GRAPH NODES
# ============================================================================

def router_node(state: State) -> State:
    """
    Routes the question to either AstraDB or Wikipedia using the LLM.
    """
    user_question = state["messages"][-1].content

    routing_prompt = (
        "You are a routing assistant. Analyze the following question and decide "
        "whether it should be answered using:\n\n"
        "1. **ASTRADB**: A vector database containing domain-specific documents "
        "(e.g., company docs, technical manuals, custom knowledge base)\n"
        "2. **WIKIPEDIA**: General knowledge encyclopedia for broad topics, "
        "historical facts, famous people, etc.\n\n"
        f"Question: {user_question}\n\n"
        "Respond with ONLY one word: either \"ASTRADB\" or \"WIKIPEDIA\"\n\n"
        "Guidelines:\n"
        "- Use ASTRADB for: domain-specific queries, internal company info, "
        "technical documentation, specialized knowledge\n"
        "- Use WIKIPEDIA for: general knowledge, famous people, historical events, "
        "scientific concepts, geography, etc."
    )

    routing_messages = [
        SystemMessage(
            content="You are a routing decision maker. Respond with only 'ASTRADB' or 'WIKIPEDIA'."
        ),
        HumanMessage(content=routing_prompt),
    ]

    response = llm.invoke(routing_messages)
    route_decision = response.content.strip().upper()

    if route_decision not in ("ASTRADB", "WIKIPEDIA"):
        route_decision = "WIKIPEDIA"

    print(f"🔀 Routing Decision: {route_decision}")

    return {
        "route_decision": route_decision,
        "messages": [AIMessage(content=f"Routing to {route_decision}...")],
    }


def astradb_retrieval_node(state: State) -> State:
    """Retrieves relevant documents from AstraDB vector store."""
    user_question = state["messages"][0].content

    print(f"📚 Searching AstraDB for: {user_question}")

    try:
        results = astra_vector_store.similarity_search(query=user_question, k=4)

        retrieved_docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": "AstraDB",
                "rank": idx,
            }
            for idx, doc in enumerate(results, 1)
        ]

        print(f"✅ Retrieved {len(retrieved_docs)} documents from AstraDB")

        return {
            "retrieved_docs": retrieved_docs,
            "messages": [
                AIMessage(content=f"Retrieved {len(retrieved_docs)} documents from AstraDB")
            ],
        }

    except Exception as e:
        print(f"❌ Error retrieving from AstraDB: {e}")
        return {
            "retrieved_docs": [],
            "messages": [AIMessage(content=f"Error retrieving from AstraDB: {e}")],
        }


def wikipedia_retrieval_node(state: State) -> State:
    """Retrieves relevant information from Wikipedia."""
    user_question = state["messages"][0].content

    print(f"🌐 Searching Wikipedia for: {user_question}")

    try:
        wiki_results = wikipedia_tool.run(user_question)

        retrieved_docs = [
            {
                "content": wiki_results,
                "metadata": {"source": "Wikipedia"},
                "source": "Wikipedia",
                "rank": 1,
            }
        ]

        print("✅ Retrieved information from Wikipedia")

        return {
            "retrieved_docs": retrieved_docs,
            "messages": [AIMessage(content="Retrieved information from Wikipedia")],
        }

    except Exception as e:
        print(f"❌ Error retrieving from Wikipedia: {e}")
        return {
            "retrieved_docs": [],
            "messages": [AIMessage(content=f"Error retrieving from Wikipedia: {e}")],
        }


def response_generator_node(state: State) -> State:
    """Generates final response using the retrieved documents."""
    user_question = state["messages"][0].content
    retrieved_docs = state["retrieved_docs"]

    if not retrieved_docs:
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but I couldn't retrieve any relevant "
                    "information to answer your question."
                )
            ]
        }

    context_parts = [
        f"[{doc['source']} - Rank {doc['rank']}]\n{doc['content']}\n"
        for doc in retrieved_docs
    ]
    context = "\n---\n".join(context_parts)

    response_prompt = (
        "Based on the following retrieved information, answer the user's question "
        "accurately and comprehensively.\n\n"
        f"User Question: {user_question}\n\n"
        f"Retrieved Information:\n{context}\n\n"
        "Instructions:\n"
        "- Provide a clear, accurate answer based on the retrieved information\n"
        "- Cite the source when relevant (AstraDB or Wikipedia)\n"
        "- If the information doesn't fully answer the question, say so\n"
        "- Be concise but complete\n\n"
        "Answer:"
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on retrieved information."
        ),
        HumanMessage(content=response_prompt),
    ]

    response = llm.invoke(messages)
    print("✨ Generated final response")

    return {"messages": [AIMessage(content=response.content)]}


# ============================================================================
# CONDITIONAL EDGE LOGIC
# ============================================================================

def route_to_retrieval(state: State) -> Literal["astradb_retrieval", "wikipedia_retrieval"]:
    """Determines which retrieval node to use based on routing decision."""
    if state.get("route_decision", "").upper() == "ASTRADB":
        return "astradb_retrieval"
    return "wikipedia_retrieval"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def build_graph():
    """
    Constructs the LangGraph routing agent.

    Flow:
      router → (conditional) → astradb_retrieval / wikipedia_retrieval
             → response_generator → END
    """
    graph = StateGraph(State)

    graph.add_node("router", router_node)
    graph.add_node("astradb_retrieval", astradb_retrieval_node)
    graph.add_node("wikipedia_retrieval", wikipedia_retrieval_node)
    graph.add_node("response_generator", response_generator_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_to_retrieval,
        {
            "astradb_retrieval": "astradb_retrieval",
            "wikipedia_retrieval": "wikipedia_retrieval",
        },
    )

    graph.add_edge("astradb_retrieval", "response_generator")
    graph.add_edge("wikipedia_retrieval", "response_generator")
    graph.add_edge("response_generator", "__end__")

    return graph.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(question: str):
    """
    Run the routing agent with a given question.

    Args:
        question: User's question to answer.

    Returns:
        Final state containing the response.
    """
    print(f"\n{'=' * 70}")
    print(f"❓ Question: {question}")
    print(f"{'=' * 70}\n")

    app = build_graph()

    final_state = app.invoke({
        "messages": [HumanMessage(content=question)],
        "route_decision": "",
        "retrieved_docs": [],
    })

    print(f"\n{'=' * 70}")
    print("💬 FINAL RESPONSE:")
    print(f"{'=' * 70}")
    print(final_state["messages"][-1].content)
    print(f"{'=' * 70}\n")

    return final_state


if __name__ == "__main__":
    # Domain-specific question → routes to AstraDB
    run_agent("What are the security best practices in our company handbook?")

    # General knowledge question → routes to Wikipedia
    # run_agent("Who is Shah Rukh Khan?")

    # Interactive mode
    # print("\n" + "=" * 70)
    # print("Interactive Mode - Enter your questions (or 'quit' to exit)")
    # print("=" * 70 + "\n")
    # while True:
    #     user_input = input("Your question: ").strip()
    #     if user_input.lower() in ("quit", "exit", "q"):
    #         print("Goodbye!")
    #         break
    #     if user_input:
    #         run_agent(user_input)