import os
import pandas as pd
from langchain.retrievers import MultiVectorRetriever
from langchain.storage.in_memory import InMemoryByteStore
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    ImageCaptionLoader,
    WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Initialize the cache
set_llm_cache(InMemoryCache())

# Define paths, constants and embedding function
directory_path = os.path.abspath("data")
text_vectorstore_path = "vectorstore/text"
image_vectorstore_path = "vectorstore/image"
web_vectorstore_path = "vectorstore/web"
multi_vectorstore_path = "vectorstore/multi"
chunk_size = 100
chunk_overlap = 10
batch_size = 100
embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
os.environ["USER_AGENT"] = "RealEstateQnA/1.0 (Python LangChain; rahulpandey@gmail.com)"

# Correct BLIP processor and model names
blip_processor_name = "Salesforce/blip-image-captioning-base"
blip_model_name = "Salesforce/blip-image-captioning-base"


def retriever_creator():
    """
    Loads documents, creates embeddings, and generates a multi-retriever.
    Handles PDF, DOCX, XLSX, and image files, and extracts data from links in DOCX.
    """

    # Text Retriever
    if os.path.exists(text_vectorstore_path):
        print("Loading existing text vectorstore...")
        text_vectorstore = Chroma(persist_directory=text_vectorstore_path, embedding_function=embedding_function)
    else:
        print("Creating new text vectorstore...")

        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory path {directory_path} is not valid or does not exist.")

        text_loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader
            # No XLSX loader here, handle it separately below
        }

        text_documents = []
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1]
            filepath = os.path.join(directory_path, filename)

            if file_extension in text_loaders:
                loader = text_loaders[file_extension](filepath)
                text_documents.extend(loader.load())
            elif file_extension == ".xlsx":
                # Load XLSX using pandas and append directly to `documents`
                df = pd.read_excel(filepath)
                text_documents.extend([Document(page_content=str(row)) for _, row in df.iterrows()])

        print(f'You have {len(text_documents)} text document(s) in your data')

        # Split, clean, and embed text documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(text_documents)

        texts = [text.page_content.replace("\n", " ") for text in texts]

        # Create and persist (or load if exists) text vectorstore
        text_vectorstore = Chroma(persist_directory=text_vectorstore_path, embedding_function=embedding_function)
        if not os.path.exists(text_vectorstore_path):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                text_vectorstore.add_texts(texts=batch)

            print("Text embeddings created and vectorstore saved.")

    text_retriever = text_vectorstore.as_retriever()

    # Image Retriever
    if os.path.exists(image_vectorstore_path):
        print("Loading existing image vectorstore...")
        image_vectorstore = Chroma(persist_directory=image_vectorstore_path, embedding_function=embedding_function)
    else:
        print("Creating new image vectorstore...")

        image_documents = []
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension in [".jpg", ".jpeg", ".png"]:
                filepath = os.path.join(directory_path, filename)
                loader = ImageCaptionLoader(filepath, blip_processor=blip_processor_name, blip_model=blip_model_name)
                image_documents.extend(loader.load())

        # Split, clean, and embed image captions
        image_texts = [doc.page_content.replace("\n", " ") for doc in image_documents]

        # Create and persist (or load if exists) image vectorstore
        image_vectorstore = Chroma(persist_directory=image_vectorstore_path, embedding_function=embedding_function)
        if not os.path.exists(image_vectorstore_path):
            for i in range(0, len(image_texts), batch_size):
                batch = image_texts[i:i + batch_size]
                image_vectorstore.add_texts(texts=batch)

            print("Image embeddings created and vectorstore saved.")

    image_retriever = image_vectorstore.as_retriever()

    # Web Retriever
    if os.path.exists(web_vectorstore_path):
        print("Loading existing web vectorstore...")
        web_vectorstore = Chroma(persist_directory=web_vectorstore_path, embedding_function=embedding_function)
    else:
        print("Creating new web vectorstore...")

        web_documents = []

        # Extract links from DOCX files
        for filename in os.listdir(directory_path):
            if filename.endswith(".docx"):
                filepath = os.path.join(directory_path, filename)
                loader = Docx2txtLoader(filepath)
                for doc in loader.load():
                    for link in doc.metadata.get("links", []):
                        try:
                            link_loader = WebBaseLoader(link)
                            web_documents.extend(link_loader.load())
                        except Exception as e:
                            print(f"Error loading link {link}: {e}")

        # Split, clean, and embed web documents
        web_texts = text_splitter.split_documents(web_documents)
        web_texts = [text.page_content.replace("\n", " ") for text in web_texts]

        # Create and persist (or load if exists) web vectorstore
        web_vectorstore = Chroma(persist_directory=web_vectorstore_path, embedding_function=embedding_function)
        if not os.path.exists(web_vectorstore_path):
            for i in range(0, len(web_texts), batch_size):
                batch = web_texts[i:i + batch_size]
                web_vectorstore.add_texts(texts=batch)

            print("Web embeddings created and vectorstore saved.")

    web_retriever = web_vectorstore.as_retriever()

    # Create an InMemoryByteStore for byte storage
    byte_store = InMemoryByteStore()

    # Multi-vectorstore to store all combined vectors
    multi_vectorstore = Chroma(persist_directory=multi_vectorstore_path, embedding_function=embedding_function)

    # Combine retrievers into a MultiVectorRetriever
    multi_retriever = MultiVectorRetriever(
        retrievers=[text_retriever, image_retriever, web_retriever],
        byte_store=byte_store,
        vectorstore=multi_vectorstore
    )

    return multi_retriever
