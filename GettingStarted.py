# This script demonstrates how to build a retrieval-augmented (RAG) question answering (QA) system using LangChain and OpenAI APIs. It performs the following steps:
# 1. Loads environment variables for API keys and LangSmith tracking.
# 2. Initializes a ChatOpenAI LLM (GPT-4o).
# 3. Loads web content from a specified URL using WebBaseLoader.
# 4. Splits the loaded documents into manageable chunks using RecursiveCharacterTextSplitter.
# 5. Generates embeddings for the document chunks with OpenAIEmbeddings.
# 6. Stores the embeddings in a FAISS vector database for efficient similarity search.
# 7. Performs a similarity search in the vector database for a given query.
# 8. Defines a prompt template for answering questions based on retrieved context.
# 9. Creates a document chain for generating answers using the LLM and the prompt.
# 10. Sets up a retriever from the vector database and creates a retrieval chain combining retrieval and answer generation.
# 11. Invokes the retrieval chain with a sample input question and prints the generated answer and the supporting context.
# Dependencies:
# - dotenv
# - langchain_openai
# - langchain_community
# - langchain_text_splitters
# - langchain.vectorstores
# - langchain.chains
# - langchain.prompts
# - langchain_core.documents
# - FAISS
# Intended for use as a demonstration or starting point for building retrieval-augmented generation (RAG) pipelines with LangChain and OpenAI.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

llm=ChatOpenAI(model="gpt-4o")
# print(llm)
# Loading the documents from the URL and splitting them into chunks
loader = WebBaseLoader("https://docs.smith.langchain.com/administration/tutorials/manage_spend");
docs = loader.load();
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

#Storing the documents in a vector store
embeddings = OpenAIEmbeddings()
vectorstoredb = FAISS.from_documents(documents, embeddings)

query="LangSmith has two usage limits: total traces and extended"
result = vectorstoredb.similarity_search(query, k=1);
# print("Som >>>> " + result[0].page_content)

prompt = ChatPromptTemplate.from_template(
    """

Answer the following question  based only on the provided context : 
<context>
{context}
</context>
"""
)
document_chain = create_stuff_documents_chain(llm, prompt=prompt)

# print(document_chain)

# p = document_chain.invoke({
#                        "input" : "LangSmith has two usage limits: total traces and extended",
#                        "context" : [Document(page_content="LangSmith has two usage limits: total traces and extended retention traces. These correspond to the two metrics we've been tracking on our usage graph.")]

#                        })

# print(p)

#Retriever uses the power of vector-DB + llm 
# where as  document_chain which uses only document chunks along with the LLM  which is time consuming and also expensive for large documents .

retriever = vectorstoredb.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)
response = retrieval_chain.invoke({"input" : "LangSmith has two usage limits: total traces and extended"})
# print("ANSWER >>>> ")
# print(response['answer'])

# print("RESPONSE >>>> ")
# print((response['context']))


# the below code demonstrates the use of LCEL
#SystemMessage - definiing how the LLM should behave
# HumanMessage - the input from the user
# AIMessage - the output from the LLM

generic_template = "Translate the following text to {language}: "
chatPrompt_Template = ChatPromptTemplate.from_messages(
    [("system", generic_template), ("user", "{text}")]
)


parser = StrOutputParser()

#LCEL - LangChain Embedding Language below
chain = chatPrompt_Template | llm | parser
print(chain.invoke({
    "language": "French",
    "text": "I love programming in Python. It is a great language for data science."
}))


