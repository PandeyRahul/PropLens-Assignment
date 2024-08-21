from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from prompt import system_prompt
from data_ingestion import retriever_creator

# Initialize LLM with Ollama (Ensure base_url is correct and model is running)
llm = ChatOllama(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Create the question-answer chain (chain that processes retrieved documents)
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Initialize the retriever using the retriever_creator function
retriever = retriever_creator()

# Validate if the retriever is correctly created
if retriever is None:
    raise ValueError("Retriever was not properly initialized. Please check retriever_creator().")

# Create the retrieval augmented generation (RAG) chain using the retriever
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Main loop for interaction
while True:
    print("Bot: Please ask me any question, or type 'exit'/'quit' to end the session.")
    question = input("You: ")
    if question.lower() in ["exit", "quit", "bye"]:
        print("See you again.")
        break

    try:
        # Invoke the RAG chain to get the answer
        results = rag_chain.invoke({"input": question})
        print("Bot: ", results["answer"])
    except Exception as e:
        print(f"An error occurred: {e}")
