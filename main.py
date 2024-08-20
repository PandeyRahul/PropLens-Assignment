from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from prompt import system_prompt
from data_ingestion import retriever_creator

# Initialize LLM and embedding function
llm = ChatOllama(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval augmented generation (RAG) chain
rag_chain = create_retrieval_chain(retriever_creator(), question_answer_chain)

while True:
    print("Bot: Please ask me any question else if you want to exit please type in exit or quit.")
    question = input("You: ")
    if question.lower() in ["exit", "quit", "bye"]:
        print("See you again.")
        break
    results = rag_chain.invoke({"input": question})
    print("Bot: ", results["answer"])
