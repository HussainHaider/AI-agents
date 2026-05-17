from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant that provides information about programming languages."),
    HumanMessage(content="What is Python?"),
]

model_response = model.invoke(messages)
messages.append(AIMessage(content=model_response.content))

print(messages)