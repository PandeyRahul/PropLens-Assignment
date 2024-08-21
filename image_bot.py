from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import io
from langchain.agents import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_ollama.chat_models import ChatOllama
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain

# Set up the language model
llm = ChatOllama(model="llava:7b", base_url="http://127.0.0.1:11434/")


class LLaVATool(BaseTool):
    name = "LLaVA"
    description = "Use this tool to process an image and generate a description, detect objects, or extract text."

    def _run(self, img_path: str) -> str:
        try:
            # Load the image
            image = Image.open(img_path).convert('RGB')

            # Convert the image to a byte array
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create a HumanMessage with the image as an additional argument
            human_message = HumanMessage(
                content="Please analyze the following image.",
                additional_kwargs={"image": img_byte_arr}
            )

            # Pass the message to the LLM agent
            response = llm.invoke([human_message])

            # Return the response text
            return response["messages"][-1].content

        except Exception as e:
            return f"Error processing the image or generating the response: {str(e)}"


# Define the tools the agent can use
tools = [
    Tool(
        name="LLaVA",
        func=LLaVATool()._run,
        description="Processes an image using LLaVA to generate descriptions, detect objects, and extract text."
    )
]

# Define the system prompt with more precise instructions
prompt = """
You are an advanced image analysis assistant. Your task is to analyze images provided by the user and generate detailed, factual descriptions based on the user's query.
You have access to the LLaVA model, which you can use to describe images, detect objects, and extract text.
When a user provides an image and a query:
1. Process the image using the LLaVA model.
2. Generate a detailed response that directly addresses the user's query based on the image.
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
vectorstore = FAISS.from_texts([" "], HuggingFaceEmbeddings())  # Initialize with an empty string

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
