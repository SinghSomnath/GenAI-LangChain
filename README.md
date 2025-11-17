Below is the versions where the repo is built so far 11/17/2025 with the Python version >>> 3.12.7


langchain                                0.3.27

langchain-chroma                         0.2.5

langchain-community                      0.3.27

langchain-core                           0.3.74

langchain-groq                           0.3.7

langchain-huggingface                    0.3.1

langchain-mcp-adapters                   0.1.9

langchain-openai                         0.3.30

langchain-text-splitters                 0.3.9

pip                                      25.1

streamlit                                1.48.1



############################################################################################################
############################################################################################################




# GenAI-LangChain
Gen Ai dev projects using LangChain lib


app_tools_agents.py >>> 

This Streamlit app creates a chatbot interface that uses LangChain agents and large language models (LLMs) to answer user queries. It integrates tools for searching the web (DuckDuckGo), Arxiv, and Wikipedia, allowing the agent to fetch and summarize information from multiple sources. The app manages chat history using Streamlit's session state and displays both user and assistant messages in a conversational format. When a user submits a prompt, the agent selects the best tool to answer the question, and its reasoning process is shown in real time using StreamlitCallbackHandler. The assistant's response is then appended to the chat history and displayed to the user.

app_FastAPI_StreamLit.py >>> 

This app combines a FastAPI backend and a Streamlit frontend to create a conversational RAG (Retrieval-Augmented Generation) system for question answering over uploaded PDF documents. Users upload PDFs via the Streamlit interface, which are processed by FastAPI: the documents are split, embedded, and stored in a vector database for semantic retrieval. The system uses a Groq language model to reformulate user questions and generate concise answers based on retrieved document context, maintaining chat history for each session. The Streamlit app allows users to upload files, enter a session ID, and ask questions, displaying both answers and chat history. Both the FastAPI server and Streamlit app run concurrently, enabling real-time document-based conversational AI.

langchain_sql_chat.py >>>  

This Streamlit app provides a conversational interface for querying a SQL database (either a local SQLite database or a user-specified MySQL database) using natural language, powered by a Groq language model. Users select the database type and enter connection details and an API key in the sidebar. The app configures the database connection (with caching for efficiency), sets up a LangChain SQL agent with a toolkit, and manages chat history in the session state. When a user submits a query, the agent interprets the question, generates and executes the appropriate SQL, and returns the answer in the chat interface. The assistant's responses and the user's queries are displayed in a conversational format, enabling interactive exploration of the database.



createDB.tf >>> 

This Terraform code provisions a secure AWS infrastructure with a VPC, public and private subnets, and all necessary networking components. It deploys a bastion host in a public subnet, allowing secure SSH access to resources in the private network using a generated key pair. An RDS MySQL database instance is created in private subnets, protected by a security group that only allows connections from the bastion host. The setup ensures the database is not publicly accessible, enhancing security, while still allowing access via an SSH tunnel through the bastion. Output values provide ready-to-use SSH and MySQL tunnel commands for connecting to the bastion host and the private RDS instance.


commands to run the terraform script createDB.tf  is below which is a precursor to setup the RDS my sql db in AWS  for langchain_sql_chat.py to use  :
----------------------------------------------------------------------------
----------------------------------------------------------------------------

1> terraform init 

>>>  This downloads provider plugins (like AWS).

2> terraform validate

>>> Check your script is valid:

3> terraform plan -out=plan.tfplan

>>> See what Terraform will do

4> terraform apply plan.tfplan

>>> Actually create resources:

5> terraform destroy

>>> When you’re done and want to remove resources

That’s the basic flow: init → validate → plan → apply → destroy.

the step num 5 will print something like 


bastion_ssh_command = "ssh -i bastion-key.pem ubuntu@<ec2-bastion-public-ip>"
mysql_tunnel_command = "ssh -i bastion-key.pem -L 3306:<rds-endpoint>:3306 ec2-user@<ec2-bastion-public-ip>"

6> Then run the batch file "setup_tunnel.bat <rds-endpoint> <ec2-bastion-public-ip>" to enable to port fowarding .

7> While the ssh tunnel remained active after doing the >>> step num 6 <<<<< run the relevant python code to connect to the mysql rds .


#########################################################################
#########################################################################
#########################################################################

Sample python commands install dependency from reuirements.txt and run any of the streamlit application (eg app_tools_agents.py)


required commands :

>pip install -r requirements.txt 


>streamlit run .\app_tools_agents.py

