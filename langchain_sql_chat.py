import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
import sqlite3
from langchain.sql_database import SQLDatabase


st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🗣️")

st.title("🗣️ LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to your SQL Database"]
selected_opt = st.sidebar.radio("Choose the DB which you want to use", options=radio_opt)

if selected_opt == radio_opt[1]:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL Password", type="password")
    mysql_db = st.sidebar.text_input("MYSQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not db_uri:
    st.info("Please enter the database information and uri!")

if not api_key:
    st.info("Please add the groq api key!")

# Initialize LLM
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama-3-8b-8192", streaming=True)
else:
    st.error("Please provide a valid Groq API key in the sidebar.")
    st.stop()

# cache resource(ttl = "2h")
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfile_path = (Path(__file__).parent/"student.db").absolute()
        print(dbfile_path)
        creator = lambda: sqlite3.connect(f"file://{dbfile_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# configure db
configure_db(db_uri)

# toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)