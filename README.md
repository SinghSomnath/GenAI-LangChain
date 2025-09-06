# GenAI-LangChain
Gen Ai dev projects using LangChain lib

This script demonstrates how to build a simple retrieval-augmented question-answering (QA) system using LangChain and OpenAI APIs. 

1> It starts by loading environment variables (such as API keys) using `dotenv`, which allows secure management of credentials. 

2> The script then explicitly sets the necessary environment variables for OpenAI and LangChain tracking, ensuring that the required services are authenticated and that tracing is enabled for debugging or monitoring purposes.

3> The core workflow begins by initializing a `ChatOpenAI` large language model (LLM) with the "gpt-4o" model. 
Next, it uses `WebBaseLoader` to fetch and load the content from a specific web page (in this case, a LangChain documentation page about managing spend). 
The loaded documents are then split into manageable chunks using `RecursiveCharacterTextSplitter`, which helps ensure that each chunk fits within the model's context window and maintains semantic coherence.

4> To enable semantic search, the script generates vector embeddings for each document chunk using `OpenAIEmbeddings`. 
These embeddings are stored in a FAISS vector database, which allows for efficient similarity search. 
When a user query is provided, the script retrieves the most relevant document chunk(s) from the vector store using a similarity search.

5> A prompt template is defined using `ChatPromptTemplate`, instructing the LLM to answer questions strictly based on the provided context. 
The `create_stuff_documents_chain` function combines the LLM and the prompt into a chain that can process documents and generate answers. 
The script then wraps this logic in a retrieval chain, which first retrieves relevant context and then passes it to the LLM for answer generation.

6> Finally, the script invokes the retrieval chain with a sample query about LangSmith's usage limits, prints the generated answer, and displays the context used for answering. 
This approach ensures that the LLM's responses are grounded in the retrieved documentation, making the answers more accurate and trustworthy.

required commands :

>pip install -r requirements.txt 

 [The command pip install -r requirements.txt is used to install all the Python packages listed in the requirements.txt file. This file typically contains a list of dependencies required for your project, with each line specifying a package name and, optionally, a version number. By running this command in your terminal, pip (the Python package installer) reads the file and installs each package, ensuring your development environment matches the project's requirements. This approach helps maintain consistency across different setups and makes it easier for others to set up the project on their own machines.]


 >streamlit run .\app_tools_agents.py


[The command `streamlit run .\app_tools_agents.py` is used to launch a Streamlit application by running the Python script named app_tools_agents.py. Streamlit is an open-source framework that allows you to quickly build and share interactive web apps for data science and machine learning projects using Python. When you execute this command in your terminal, Streamlit starts a local web server and opens your app in a new browser tab. This makes it easy to visualize data, interact with models, and create user-friendly interfaces without needing to write traditional front-end code. The LangChain before the script name specifies that the file is located in the current directory.]



commands to run the terraform script  createDB.tf or other .tf files are below :
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
