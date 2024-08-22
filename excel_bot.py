import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore")


def process_excel_file(file_path):
    excel_file = pd.ExcelFile(file_path)
    documents = []
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        text = df.to_string(index=False)
        documents.append({
            "page_content": f"Sheet Name: {sheet_name}\n{text}",
            "metadata": {"source": file_path, "sheet": sheet_name}
        })
    return documents


def load_and_chunk_excel_files(file_paths):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for file_path in file_paths:
        processed_docs = process_excel_file(file_path)
        for doc in processed_docs:
            chunks = text_splitter.split_text(doc["page_content"])
            for chunk in chunks:
                documents.append({
                    "page_content": chunk,
                    "metadata": doc["metadata"]
                })
    return documents


def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings()
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def create_retrieval_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


def handle_excel_query(query, file_paths):
    documents = load_and_chunk_excel_files(file_paths)
    vectorstore = create_faiss_index(documents)
    qa_chain = create_retrieval_qa_chain(vectorstore)
    response = qa_chain.invoke({"query": query})
    return response["result"]