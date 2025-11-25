from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model= "qwen3:4b")


template = """
you are and critic answering questions about a pizza restaurant

here are some reviews: {reviews}

here are some questions: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n----------")
    question = input("Ask your question (q to quit):")
    print("\n\n")
    if question == "q":
        break

    reviews = retriver.invoke(question)
    result = chain.invoke({"reviews":reviews,"question":question})

    print(result)