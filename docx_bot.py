import docx
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
import nltk

# Ensure the NLTK sentence tokenizer is downloaded
nltk.download('punkt')

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


# Function to split text into sentences
def split_text_into_sentences(text):
    return nltk.tokenize.sent_tokenize(text)


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


# Function to load and chunk the documents with more granular splitting
def load_and_chunk_documents(documents):
    chunked_documents = []
    for doc in documents:
        # Split the text into sentences
        sentences = split_text_into_sentences(doc["page_content"])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)

        # Join sentences into chunks that fit the token limit
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 150:
                # When the chunk exceeds the limit, add it to the list and start a new chunk
                chunked_documents.append({
                    "page_content": current_chunk.strip(),
                    "metadata": doc["metadata"]
                })
                current_chunk = sentence
            else:
                current_chunk += " " + sentence

        # Add any remaining text as the last chunk
        if current_chunk:
            chunked_documents.append({
                "page_content": current_chunk.strip(),
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


# Function to handle the query via FastAPI route
def handle_docx_query(query, docx_paths):
    # Step 1: Process docx files
    documents = process_docx_files(docx_paths)

    # Step 2: Chunk the documents with more granular splitting
    chunked_documents = load_and_chunk_documents(documents)

    # Step 3: Create FAISS index
    vectorstore = create_faiss_index(chunked_documents)

    # Step 4: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Step 5: Handle the query
    response = qa_chain.invoke({"query": query})
    return response["result"]
