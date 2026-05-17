from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain {topic} in simple terms."),
    # SystemMessage(content="You are a helpful {domain} expert."), # this will not work because the template expects a tuple with the role and content, not a SystemMessage object
    # HumanMessage(content="Explain {topic} in simple terms."),
])

prompt = chat_template.invoke({
    'domain': 'machine learning',
    'topic': 'neural networks'
})

print(prompt)