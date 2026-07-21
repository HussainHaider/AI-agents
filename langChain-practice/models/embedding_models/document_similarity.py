from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Babar Azam is a Pakistani cricketer known for his elegant batting and consistent performances across formats.",
    "Shaheen Shah Afridi is a Pakistani fast bowler famous for his pace, swing, and wicket-taking ability.",
    "Mohammad Rizwan is a Pakistani wicketkeeper-batsman known for his hard work, consistency, and match-winning innings.",
    "Wasim Akram is a legendary Pakistani fast bowler regarded as one of the greatest swing bowlers in cricket history.",
    "Imran Khan is a former Pakistani captain who led Pakistan to victory in the 1992 Cricket World Cup.",
    "Inzamam-ul-Haq was a Pakistani batsman known for his calm temperament and powerful stroke play.",
    "Saeed Anwar was a stylish Pakistani opener famous for his fluent batting and high scores in ODI cricket.",
    "Younis Khan is a former Pakistani batsman who holds the record for the most Test runs by a Pakistani player.",
    "Shahid Afridi was a Pakistani all-rounder known for his explosive batting and entertaining style of play.",
    "Abdul Qadir was a legendary Pakistani leg-spinner who played a major role in popularizing leg-spin bowling.",
]

query = "tell me about Afridi"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]
print(scores)

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)
