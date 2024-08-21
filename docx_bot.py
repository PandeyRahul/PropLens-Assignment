import os
import docx
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Function to extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)


# Function to extract data from links within the text
def extract_and_append_link_data(text):
    soup = BeautifulSoup(text, 'html.parser')
    links = [link.get('href') for link in soup.find_all('a')]
    link_texts = []

    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                page_content = BeautifulSoup(response.content, 'html.parser').get_text()
                link_texts.append(page_content)
            else:
                link_texts.append(f"Failed to retrieve content from {link}")
        except requests.exceptions.RequestException as e:
            link_texts.append(f"Error retrieving content from {link}: {e}")

    return "\n".join(link_texts)


# Function to process docx files, including handling links
def process_docx_files(docx_paths):
    documents = []
    for docx_path in docx_paths:
        text = extract_text_from_docx(docx_path)

        # If the document contains links, extract and append data from those links
        if "Project links.docx" in docx_path:
            link_data = extract_and_append_link_data(text)
            text += "\n\n" + link_data

        documents.append({
            "page_content": text,
            "metadata": {"source": docx_path}
        })
    return documents


# Function to load and chunk the documents
def load_and_chunk_documents(documents):
    # Reduce chunk size to avoid token length errors
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunked_documents = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["page_content"])
        for chunk in chunks:
            chunked_documents.append({
                "page_content": chunk,
                "metadata": doc["metadata"]
            })

    return chunked_documents


# Function to create a FAISS index from the documents
def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings()
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


# Function to create a RetrievalQA chain
def create_retrieval_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


# Main function
def main(docx_paths):
    # Step 1: Process docx files
    documents = process_docx_files(docx_paths)

    # Step 2: Chunk the documents
    chunked_documents = load_and_chunk_documents(documents)

    # Step 3: Create FAISS index
    vectorstore = create_faiss_index(chunked_documents)

    # Step 4: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Continuous interaction loop
    while True:
        print("\nBot: Please ask any question related to the documents or type 'exit' to quit.")
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        response = qa_chain.invoke({"query": query})
        print("Bot: ", response["result"])


# List of .docx file paths to be processed
docx_paths = []
for dirpath, dirnames, filenames in os.walk("data"):
    for filename in [f for f in filenames if f.endswith(".docx")]:
        docx_paths.append(os.path.join(os.path.abspath("data"), filename))

# Run the main function
if __name__ == "__main__":
    main(docx_paths)
