import fitz  # PyMuPDF for PDF handling
import pytesseract  # OCR library
from PIL import Image  # Image processing
from io import BytesIO  # Handling byte streams for images
from langchain_community.document_loaders import PDFPlumberLoader, PyMuPDFLoader  # PDF loaders
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embeddings
from langchain_community.vectorstores.faiss import FAISS  # FAISS for vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitter for chunking
from langchain.chains import RetrievalQA  # QA chain for retrieval
from langchain_ollama.llms import OllamaLLM  # LLaMA model integration
import warnings  # Warning handling

# Suppress all warnings to avoid cluttering the output
warnings.filterwarnings("ignore")

# Configure Tesseract OCR path for image text extraction
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Function to extract images from PDFs using PyMuPDF
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
    if use_pymupdf:
        loader = PyMuPDFLoader(pdf_path)
    else:
        loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    extracted_text = " ".join([doc.page_content for doc in documents])
    images = extract_images_pymupdf(pdf_path)
    if images:
        ocr_text = extract_text_from_images(images)
        extracted_text += " " + ocr_text
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
        chunks = text_splitter.split_text(document["page_content"])
        for chunk in chunks:
            documents.append({
                "page_content": chunk,
                "metadata": document["metadata"]
            })
    return documents


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
def handle_pdf_query(query, pdf_paths):
    # Step 1: Load and process PDFs
    documents = load_into_langchain(pdf_paths)

    # Step 2: Create FAISS index
    vectorstore = create_faiss_index(documents)

    # Step 3: Create RetrievalQA chain
    qa_chain = create_retrieval_qa_chain(vectorstore)

    # Step 4: Handle the query
    response = qa_chain({"query": query})
    return response["result"]
