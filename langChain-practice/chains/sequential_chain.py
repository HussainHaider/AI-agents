from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write a 5 line summary about the following text. \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic": "the impact of AI on society"})

print("result")
print(result)

print(chain.get_graph().print_ascii())
