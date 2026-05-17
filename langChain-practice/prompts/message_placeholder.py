from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
#chat template with a placeholder for messages
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "{query}"),
])

chat_history = []
#load chat history from a file or database
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

#create prompt with the chat history and a new query
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': HumanMessage(content="How is my refund?")
})