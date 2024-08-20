import os
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths, constants and embedding function
directory_path = os.path.abspath("data")
vectorstore_path = "vectorstore"
chunk_size = 100
chunk_overlap = 10
batch_size = 100
embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


def retriever_creator():
    """
    Function that loads document, create embeddings, and generates a retriever
    :return:
    """
    # Load and process documents
    if os.path.exists(vectorstore_path):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
    else:
        print("Creating new vectorstore...")

        # Validate directory
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory path {directory_path} is not valid or does not exist.")

        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            # Add more loaders for other file types if needed
        }

        documents = []
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[1]
            if file_extension in loaders:
                filepath = os.path.join(directory_path, filename)
                loader = loaders[file_extension](filepath)
                documents.extend(loader.load())

        print(f'You have {len(documents)} document(s) in your data')
        print(f'There are {len(documents[0].page_content)} characters in your document')

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        # Extract text content from Document objects and replace newlines
        texts = [text.page_content.replace("\n", " ") for text in texts]

        # Create Chroma vectorstore and add texts in batches
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectorstore.add_texts(texts=batch)

        print("Embeddings created and vectorstore saved.")

    # Use the loaded or newly created vectorstore
    retriever = vectorstore.as_retriever()

    return retriever
