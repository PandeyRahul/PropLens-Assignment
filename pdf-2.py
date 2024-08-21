import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from langchain_community.document_loaders import PDFPlumberLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Function to extract images using PyMuPDF (fitz)
def extract_images_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
    doc.close()
    return images


# Function to process images and extract text using OCR
def extract_text_from_images(images):
    image_texts = []
    for image in images:
        image_text = pytesseract.image_to_string(image)
        image_texts.append(image_text)
    return " ".join(image_texts)


# Function to load and process a single PDF using LangChain's loaders
def process_pdf(pdf_path, use_pymupdf=True):
    # print(f"Processing {pdf_path}")

    # Load the document and extract text content using the appropriate loader
    if use_pymupdf:
        loader = PyMuPDFLoader(pdf_path)  # Use PyMuPDFLoader for text extraction
    else:
        loader = PDFPlumberLoader(pdf_path)  # Use PDFPlumberLoader for text extraction

    documents = loader.load()

    # Extract text from the entire PDF document
    extracted_text = " ".join([doc.page_content for doc in documents])

    # Extract images and run OCR to get text from images
    images = extract_images_pymupdf(pdf_path)
    if images:
        ocr_text = extract_text_from_images(images)
        extracted_text += " " + ocr_text  # Combine extracted text with OCR results

    # Return the combined text with metadata
    return {
        "page_content": extracted_text,
        "metadata": {"source": pdf_path}
    }


# Function to load documents into LangChain with metadata and split text into chunks
def load_into_langchain(pdf_paths):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf_path in pdf_paths:
        document = process_pdf(pdf_path)
        # Split the text into chunks
        chunks = text_splitter.split_text(document["page_content"])
        for chunk in chunks:
            documents.append({
                "page_content": chunk,
                "metadata": document["metadata"]
            })
    return documents


# Function to create a FAISS index from the documents
def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings()  # Replace with your embedding model if needed
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


# Function to create a RetrievalQA chain
def create_retrieval_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # Or "stuff", "refine", etc. depending on your use case
        retriever=vectorstore.as_retriever(),
        return_source_documents=True  # Optional: Return source docs with the response
    )
    return qa_chain


# Main function
def main(pdf_paths):
    # Step 1: Load and process PDFs
    documents = load_into_langchain(pdf_paths)

    # Step 2: Create FAISS index
    vectorstore = create_faiss_index(documents)

    # Step 3: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Test the system with a query
    query = "What are the property tax rates for owner-occupied residential properties?"
    response = qa_chain({"query": query})
    print(response["result"])


# List of PDF paths
pdf_paths = []
for dirpath, dirnames, filenames in os.walk("data"):
    for filename in [f for f in filenames if f.endswith(".pdf")]:
        pdf_paths.append(os.path.join(os.path.abspath("data"), filename))

if __name__ == "__main__":
    main(pdf_paths)
