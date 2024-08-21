from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import pytesseract
from langchain.agents import Tool
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain


class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    description = "Generates a simple caption describing the image provided."

    def _run(self, img_path: str) -> str:
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            return f"Error opening image: {e}"

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        try:
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            inputs = processor(image, return_tensors='pt').to(device)
            output = model.generate(**inputs, max_new_tokens=20)

            caption = processor.decode(output[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            return f"Error generating caption: {e}"


class ObjectDetectionTool(BaseTool):
    name = "Object Detector"
    description = "Detects and identifies objects in the provided image."

    def _run(self, img_path: str) -> str:
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            return f"Error opening image: {e}"

        try:
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = ""
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections += f'[x1:{int(box[0])}, y1:{int(box[1])}, x2:{int(box[2])}, y2:{int(box[3])}]'
                detections += f' {model.config.id2label[int(label)]} Confidence: {float(score):.2f}\n'

            return detections

        except Exception as e:
            return f"Error detecting objects: {e}"


class OCRTool(BaseTool):
    name = "Text Extractor"
    description = "Extracts and returns any text found in the image."

    def _run(self, img_path: str) -> str:
        try:
            image = Image.open(img_path).convert('RGB')
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text: {e}"


class EnhancedImageCaptionTool(BaseTool):
    name = "Enhanced Image Captioner"
    description = "Combines OCR and image captioning to generate a detailed description of the image."

    def _run(self, img_path: str) -> str:
        try:
            # First, extract text using OCR
            image = Image.open(img_path).convert('RGB')
            extracted_text = pytesseract.image_to_string(image)

            # Generate a basic caption
            model_name = "Salesforce/blip-image-captioning-large"
            device = "cpu"

            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

            inputs = processor(image, return_tensors='pt').to(device)
            output = model.generate(**inputs, max_new_tokens=20)

            basic_caption = processor.decode(output[0], skip_special_tokens=True)

            # Combine the OCR results with the basic caption for a more detailed description
            combined_caption = f"OCR extracted text: {extracted_text.strip()} | Image caption: {basic_caption.strip()}"

            return combined_caption

        except Exception as e:
            return f"Error generating enhanced caption: {e}"


# Set up the language model
llm = ChatOllama(model="llama3.1:8b", base_url="http://127.0.0.1:11434/")

# Define the tools the agent can use
tools = [
    Tool(
        name="Image Captioner",
        func=ImageCaptionTool()._run,
        description="Generates a simple caption describing the image provided."
    ),
    Tool(
        name="Object Detector",
        func=ObjectDetectionTool()._run,
        description="Detects and identifies objects in the provided image."
    ),
    Tool(
        name="Text Extractor",
        func=OCRTool()._run,
        description="Extracts and returns any text found in the image."
    ),
    Tool(
        name="Enhanced Image Captioner",
        func=EnhancedImageCaptionTool()._run,
        description="Combines OCR and image captioning to generate a detailed description of the image."
    ),
]

# Define the system prompt with more precise instructions
prompt = """
You are an advanced image analysis assistant. Your task is to analyze images provided by the user and generate detailed, factual descriptions based on the user's query.
Your tools include:
- Image Captioner: Generates a basic caption describing the image.
- Object Detector: Detects and identifies objects in the image.
- Text Extractor: Extracts any text present in the image.
- Enhanced Image Captioner: Combines OCR (text extraction) and image captioning to generate a detailed, contextual description.
When a user provides an image and a query:
1. Determine the most relevant tool(s) to use based on the query.
2. Use the selected tool(s) to analyze the image.
3. Combine the results to provide a detailed response directly addressing the user's query.
Avoid any irrelevant information, assumptions, or speculation. Stick to factual, observable details from the image.
If the image cannot be processed, clearly and concisely communicate this issue to the user.
"""

# Initialize conversation memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Create a simple FAISS vectorstore for chat history
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts([" "], embeddings)  # Initialize with an empty string

# Create the QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=conversational_memory,
    verbose=True
)

# Create the agent with the defined prompt
system_message = SystemMessage(content=prompt)
agent = create_react_agent(llm, tools, state_modifier=system_message)


# Function to handle image description requests using the LLM agent
def handle_image_description(image_path, user_question):
    # Combine the user question with image path information
    user_input_with_image = f"{user_question}. Here is the image path: {image_path}"

    # Use the agent to handle the input
    response = agent.invoke({"messages": [HumanMessage(content=user_input_with_image)]})

    # Extract and return the final content from the response
    return response["messages"][-1].content


# Sample image path and user input
image_path = "./data/Tembusu grand 1 Bed + Study unit plan.png"
user_question = input("Please ask your question.\n")

# Handle the image description request using the LLM agent
response = handle_image_description(image_path, user_question)
print(response)
