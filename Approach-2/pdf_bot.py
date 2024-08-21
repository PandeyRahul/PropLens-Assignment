import os
from dotenv.main import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# Initialize LLM with Ollama (Ensure base_url is correct and model is running)
llm = ChatOllama(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")


def create_pdf_agent(directory_name, llm, query):
    # Get the absolute path to the directory
    directory_path = os.path.abspath(directory_name)

    # Check if the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(
            f"Directory path {directory_path} is not valid or does not exist."
        )

    # Load PDF documents from the directory
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define the system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval augmented generation (RAG) chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": query})
    return results["answer"]


query = input("Please ask me a question?")

print(create_pdf_agent("data", llm, query))
