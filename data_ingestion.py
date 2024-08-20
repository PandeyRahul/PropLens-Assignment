import os
import pandas as pd
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

# Define paths, constants and embedding function
directory_path = os.path.abspath("data")
vectorstore_path = "vectorstore"
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
    Loads documents, creates embeddings, and generates a retriever.
    Handles PDF, DOCX, XLSX, and image files, and extracts data from links in DOCX.
    """

    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
    else:
        print("Creating new vectorstore...")

        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory path {directory_path} is not valid or does not exist.")

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".jpg": lambda x: ImageCaptionLoader(x, blip_processor=blip_processor_name, blip_model=blip_model_name),
            ".jpeg": lambda x: ImageCaptionLoader(x, blip_processor=blip_processor_name, blip_model=blip_model_name),
            ".png": lambda x: ImageCaptionLoader(x, blip_processor=blip_processor_name, blip_model=blip_model_name)
        }

        documents = []
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1]
            filepath = os.path.join(directory_path, filename)

            if file_extension in loaders:
                loader = loaders[file_extension](filepath)
                loaded_docs = loader.load()
            elif file_extension == ".xlsx":
                # Load XLSX using pandas
                df = pd.read_excel(filepath)
                loaded_docs = [Document(page_content=str(row)) for _, row in df.iterrows()]
            else:
                print(f"Skipping unsupported file: {filename}")
                continue

            # Extract links from DOCX files and load them
            if file_extension == ".docx":  # Correct indentation
                for doc in loaded_docs:
                    for link in doc.metadata.get("links", []):
                        try:
                            link_loader = WebBaseLoader(link)
                            documents.extend(link_loader.load())
                        except Exception as e:
                            print(f"Error loading link {link}: {e}")

            documents.extend(loaded_docs)

        print(f'You have {len(documents)} document(s) in your data')

        # Split, clean, and embed documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        texts = [text.page_content.replace("\n", " ") for text in texts]

        # Create and persist vectorstore
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectorstore.add_texts(texts=batch)

        print("Embeddings created and vectorstore saved.")

    retriever = vectorstore.as_retriever()
    return retriever


retriever_creator()
