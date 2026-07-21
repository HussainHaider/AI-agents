from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = ["Karachi is the heart of Pakistan", "Islamabad is the captial of Pakistan"]

vector = embedding.embed_documents(documents)

print(str(vector))
