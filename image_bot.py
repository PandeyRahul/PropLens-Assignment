import pytesseract
from PIL import Image
import os
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Function to perform OCR on an image and extract text
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)  # Extract text using OCR
    return text


# Function to process all images and create text-based documents
def process_images(image_paths):
    documents = []
    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        metadata = {"source": image_path}
        documents.append({
            "page_content": text,
            "metadata": metadata
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
    llm = OllamaLLM(model="llava:7b", base_url="http://127.0.0.1:11434/")  # Initialize the LLaMA model

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Using map_reduce chain type
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # Return source documents with the response
    )
    return qa_chain


# Main function
def main(image_paths):
    # Step 1: Process images and extract text
    documents = process_images(image_paths)

    # Step 2: Create FAISS index
    vectorstore = create_faiss_index(documents)

    # Step 3: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Continuous interaction loop
    while True:
        print("\nBot: Please ask any question related to the images or type 'exit' to quit.")
        query = input("You: ")

        # Exit condition
        if query.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        # Process the query and provide the response
        response = qa_chain({"query": query})
        print("Bot: ", response["result"])


# List of image paths to be processed
image_paths = []
for dirpath, dirnames, filenames in os.walk("data"):
    for filename in [f for f in filenames if f.endswith((".jpeg", ".jpg", ".png"))]:
        image_paths.append(os.path.join(os.path.abspath("data"), filename))

# Run the main function
if __name__ == "__main__":
    main(image_paths)
