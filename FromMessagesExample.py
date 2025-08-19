from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Define the template with MessagesPlaceholder for chat history
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot named {name}."),
    MessagesPlaceholder("chat_history"),
    ("human", "{user_input}")
])

# Define a sample chat history
chat_history = [
    HumanMessage(content="What’s the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

# Invoke the template with inputs, including chat history
prompt_value = template.invoke({
    "name": "Bob",
    "chat_history": chat_history,
    "user_input": "What about Spain?"
})

# Print the result
print(prompt_value)