from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
import requests
from typing import Annotated
import json


# tool create
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()


@tool
def convert(
    base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_value * conversion_rate


# print(convert.args)

# print(get_conversion_factor.invoke({"base_currency": "USD", "target_currency": "PKR"}))

# print(convert.invoke({"base_currency_value": 10, "conversion_rate": 277.6957}))

llm = ChatOpenAI()

messages = [
    HumanMessage(
        "What is the conversion factor between PKR and USD, and based on that can you convert 1000 PKR to USD"
    )
]

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

ai_message = llm_with_tools.invoke(messages)
print(f"AI Message: {ai_message}")
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    # execute the 1st tool and get the value of conversion rate
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        # append this tool message to messages list
        messages.append(tool_message1)
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call["name"] == "convert":
        # fetch the current arg
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

print(f"Final Response: {llm_with_tools.invoke(messages).content}")