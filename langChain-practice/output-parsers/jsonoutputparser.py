from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me name, age, city about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.format(topic="the solar system")
# result = model.invoke(prompt)
# parsed_result = parser.parse(result.content)

# print(parsed_result)
# print(type(parsed_result))

chain = template | model | parser
result = chain.invoke({"topic": "the solar system"})

print(result)
