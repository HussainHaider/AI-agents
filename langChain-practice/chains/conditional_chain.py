from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)

huggingFace_model = ChatHuggingFace(llm=llm)

openAI_model = ChatOpenAI()

str_parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# As LLM gives a text output we need to parse it into structured format so that we can use it for branching
parser = PydanticOutputParser(pydantic_object=Feedback)

main_prompt = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

classifier_chain = main_prompt | openAI_model | parser

pos_feedback_prompt = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

neg_feedback_prompt = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branching_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", pos_feedback_prompt | huggingFace_model | str_parser),
    (lambda x: x.sentiment == "negative", neg_feedback_prompt | huggingFace_model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branching_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()