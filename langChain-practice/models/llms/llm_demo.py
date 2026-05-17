from langchain_openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result1 = llm.invoke("What is the capital of France?")

result2 = llm.invoke("What is the capital of Pakistan?")

print(result1)
print(result2)