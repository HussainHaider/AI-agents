from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = ["Karachi is the heart of Pakistan", "Islamabad is the captial of Pakistan"]

result = embedding.embed_documents(documents)

print(str(result))