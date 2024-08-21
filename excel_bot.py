import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Function to load and process Excel sheets into a text format
def process_excel_file(file_path):
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)
    documents = []

    # Iterate over each sheet in the Excel file
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        # Convert the dataframe to a text format
        text = df.to_string(index=False)
        # Add the text along with its metadata
        documents.append({
            "page_content": f"Sheet Name: {sheet_name}\n{text}",
            "metadata": {"source": file_path, "sheet": sheet_name}
        })

    return documents


# Function to load and chunk data from multiple Excel files
def load_and_chunk_excel_files(file_paths):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Process each Excel file
    for file_path in file_paths:
        processed_docs = process_excel_file(file_path)
        # Chunk the text data
        for doc in processed_docs:
            chunks = text_splitter.split_text(doc["page_content"])
            for chunk in chunks:
                documents.append({
                    "page_content": chunk,
                    "metadata": doc["metadata"]
                })

    return documents


# Function to create a FAISS index from the documents
def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings()  # Initialize the embeddings model
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)  # Create FAISS index
    return vectorstore


# Function to create a RetrievalQA chain
def create_retrieval_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")  # Initialize the LLaMA model

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Using map_reduce chain type
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # Return source documents with the response
    )
    return qa_chain


# Main function
def main(file_paths):
    # Step 1: Load and process Excel files
    documents = load_and_chunk_excel_files(file_paths)

    # Step 2: Create FAISS index
    vectorstore = create_faiss_index(documents)

    # Step 3: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Continuous interaction loop
    while True:
        print("\nBot: Please ask any question related to the Excel sheets or type 'exit' to quit.")
        query = input("You: ")

        # Exit condition
        if query.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        # Process the query and provide the response
        response = qa_chain({"query": query})
        print("Bot: ", response["result"])


# List of Excel file paths to be processed
# List of PDF paths to be processed
excel_paths = []
for dirpath, dirnames, filenames in os.walk("data"):
    for filename in [f for f in filenames if f.endswith(".xlsx")]:
        excel_paths.append(os.path.join(os.path.abspath("data"), filename))

# Run the main function
if __name__ == "__main__":
    main(excel_paths)
