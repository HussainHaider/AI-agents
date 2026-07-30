from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import requests


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# print(multiply.invoke({"a": 3, "b": 4}))

# tool binding
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([multiply])

# message1 = HumanMessage(content="Hi, how are you?")
# response1 = llm_with_tools.invoke([message1])
# print(response1)

# tool calling
query = HumanMessage(content="Multiply 3 and 10")
messages = [query]
response = llm_with_tools.invoke(messages)
messages.append(response)
# print(f"Response: {response}")
response_with_tools = response.tool_calls

# tool Execution
if response_with_tools:
    tool_name = response_with_tools[0]["name"]
    tool_args = response_with_tools[0]["args"]
    if tool_name == "multiply":
        result = multiply.invoke(tool_args)
        print(f"Tool Execution Result: {result}")

tool_message = multiply.invoke(response_with_tools[0])
messages.append(tool_message)
print(f"Tool Message: {tool_message}")
print(f"Messages: {messages}")

llm_response = llm_with_tools.invoke(messages)
print(f"LLM Response: {llm_response.content}")