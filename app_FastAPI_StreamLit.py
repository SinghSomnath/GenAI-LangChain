import os
import streamlit as st
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import List
import uvicorn
import requests
from dotenv import load_dotenv
import asyncio

load_dotenv()

# FastAPI app setup
app = FastAPI()

# Store for session histories
session_store = {}

# Global variables for vector store
documents = []
vectorstore = None
retriever = None
conversational_rag_chain = None

class QuestionInput(BaseModel):
    session_id: str
    question: str

@app.post("/upload_pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    global documents, vectorstore, retriever, conversational_rag_chain
    
    try:
        documents = []
        for file in files:
            temppdf = f"./temp_{file.filename}"
            with open(temppdf, "wb") as f:
                f.write(await file.read())
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
            os.remove(temppdf)

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()
            
            # Initialize RAG chain
            llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Gemma2-9b-It")
            
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in session_store:
                    session_store[session_id] = ChatMessageHistory()
                return session_store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, 
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            return JSONResponse(content={"status": "success", "message": "PDFs processed successfully"})
        else:
            return JSONResponse(content={"status": "error", "message": "No documents found"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})

@app.post("/ask_question")
async def ask_question(input: QuestionInput):
    try:
        if not conversational_rag_chain:
            return JSONResponse(content={"status": "error", "message": "RAG chain not initialized. Upload PDFs first."})
        
        response = conversational_rag_chain.invoke(
            {"input": input.question},
            config={"configurable": {"session_id": input.session_id}}
        )
        
        return JSONResponse(content={
            "status": "success",
            "answer": response['answer'],
            "chat_history": [str(msg) for msg in session_store[input.session_id].messages]
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})

# Streamlit app
def run_streamlit():
    st.title("Conversational RAG with PDF uploads and chat history")
    st.write("This app allows you to upload PDF documents and ask questions about their content.")

    api_key = st.text_input("Enter your Groq API Key", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        session_id = st.text_input("Enter a session ID to maintain chat history", value="default_session")
    
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
        if uploaded_files:
            files = [("files", (file.name, file.getvalue(), "application/pdf")) for file in uploaded_files]
            response = requests.post("http://localhost:8000/upload_pdf", files=files)
        
            if response.status_code == 200:
                st.success("PDFs uploaded and processed successfully!")
            else:
                st.error(f"Error uploading PDFs: {response.json().get('message', 'Unknown error')}")

        user_input = st.text_input("Your question:")
        if user_input:
            response = requests.post(
            "http://localhost:8000/ask_question",
            json={"session_id": session_id, "question": user_input}
            )

            if response.status_code == 200:
                data = response.json()
                if 'answer' in data:
                    st.success(f"Assistant: {data['answer']}")
                    st.write("Chat History:", data['chat_history'])
                else:
                    st.error(f"No answer found in the response. {response.json()}")
            else:
                st.error(f"Error getting response: {response.json().get('message', 'Unknown error')}")

# Run both FastAPI and Streamlit
if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    import threading
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Run Streamlit
    run_streamlit()