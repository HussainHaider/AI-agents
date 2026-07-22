import warnings

from langchain_core._api import LangChainDeprecationWarning

# TextLoader is in a transitional state: both langchain_classic and
# langchain_community currently emit deprecation warnings that point at each
# other. The loader itself works fine — silence the noise for now.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-community.*being sunset.*")

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=["poem"],
)

loader = TextLoader("cricket.txt", encoding="utf-8")

loaded_documents = loader.load()

print(f"Loaded {len(loaded_documents)} documents.")
print(f"Type of loaded_documents: {type(loaded_documents)}")
print(f"Type of first document: {type(loaded_documents[0])}")
print(f"Metadata of first document: {loaded_documents[0].metadata}")

chain = prompt | model | StrOutputParser()

print("Summary of the poem:")
for doc in loaded_documents:
    summary = chain.invoke({"poem": doc.page_content})
    print(summary)
