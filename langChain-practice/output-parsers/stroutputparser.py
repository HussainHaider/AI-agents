from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# prompt 1
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

# prompt 2
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=["text"],
)

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "the impact of AI on society"})

print("result")
print(result)