from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# LangChain 1.x completely removed output_parsers from the core package — the directory doesn't even exist. 
# StructuredOutputParser has been superseded by JsonOutputParser with a Pydantic schema. 
# hece this code will not run with the latest version of LangChain. You can either downgrade to an older version of LangChain or 
# rewrite the code using the new JsonOutputParser and Pydantic schema approach.
# from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="name", description="The name of the person"),
    ResponseSchema(name="age", description="The age of the person"),
    ResponseSchema(name="city", description="The city where the person lives")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me name, age, city about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "the solar system"})
print(result)