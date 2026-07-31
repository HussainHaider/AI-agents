from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
import requests

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}"

    response = requests.get(url)

    return response.json()


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Step 2: Define the standard ReAct prompt inline (same as hwchase17/react)
prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(llm, tools=[search_tool, get_weather_data], prompt=prompt)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)

# Step 5: Invoke
response = agent_executor.invoke(
    {"input": "Find the capital of Punjab Pakistan, then find it's current weather condition"}
)

print(response)
