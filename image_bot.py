import pytesseract
from PIL import Image
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


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


def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings()
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore


def create_retrieval_qa_chain(vectorstore):
    llm = OllamaLLM(model="llava:7b", base_url="http://127.0.0.1:11434/")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


def handle_image_query(query, image_paths):
    documents = process_images(image_paths)
    vectorstore = create_faiss_index(documents)
    qa_chain = create_retrieval_qa_chain(vectorstore)
    response = qa_chain.invoke({"query": query})
    return response["result"]
