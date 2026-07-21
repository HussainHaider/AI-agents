from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-72B-Instruct", task="text-generation")

model = ChatHuggingFace(llm=llm)


class PersonInfo(BaseModel):
    name: str = Field(description="The name of the person")
    age: str = Field(description="The age of the person")
    city: str = Field(description="The city where the person lives")


parser = PydanticOutputParser(pydantic_object=PersonInfo)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"place": "sri lankan"})
print(result)
