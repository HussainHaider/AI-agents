from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

hf_model = ChatHuggingFace(llm=llm)

openai_model = ChatOpenAI()

notes_prompt = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

quiz_prompt = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

merge_prompt = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': notes_prompt | hf_model | parser,
    'quiz': quiz_prompt | openai_model | parser
})

merge_chain = merge_prompt | hf_model | parser

chain = parallel_chain | merge_chain

text = """Artificial Intelligence (AI) is a branch of 
computer science that aims to create machines capable of performing 
tasks that typically require human intelligence. 
AI encompasses various subfields, including machine learning, 
natural language processing, and computer vision. 
The applications of AI are vast, ranging from 
virtual assistants and autonomous vehicles to 
healthcare diagnostics and financial analysis. 
As AI continues to evolve, it has the potential to 
revolutionize industries and improve our daily lives, 
but it also raises ethical concerns regarding privacy, 
job displacement, and decision-making transparency."""

result = chain.invoke({'text': text})

print(result)
print(chain.get_graph().print_ascii())