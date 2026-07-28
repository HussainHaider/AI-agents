from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # v1.x API: instantiate, then fetch. Returns a FetchedTranscript object.
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])

    # to_raw_data() gives a list of {"text": ..., "start": ..., "duration": ...} dicts
    transcript_list = fetched_transcript.to_raw_data()

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])
print(f"Number of chunks: {len(chunks)}")

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")  # k = top results
# retriever.invoke('What is deepmind')

# Step 3 - Augmentation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

# Step 4 - Generation
response = llm.invoke(final_prompt)
print(f"Response: {response}")


# Building a chain for the entire process
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

chain = parallel_chain | prompt | llm | parser
chain_output = chain.invoke("Can you summarize the video?")
print(f"Chain Output: {chain_output}")
