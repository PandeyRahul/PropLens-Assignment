# Real Estate QnA System
## Overview
This project is designed to create a Real Estate QnA system capable of extracting and providing accurate answers from various data sources such as PDFs, DOCX files, Images, and Excel sheets. The system leverages state-of-the-art techniques like Large Language Models (LLMs), FAISS for vector storage, and OCR for image processing. The system is exposed via a FastAPI-based API, making it easy to integrate into other applications.

# Assignment Details
## Objective:
The goal of this assignment is to design and implement a QnA system that can fetch information related to real estate projects from a given dataset. The project utilizes modern technologies like LLMs, RAG (Retrieval-Augmented Generation), and Generative AI.

## Dataset:
The dataset comprises various documents such as PDFs, DOCX files, images, and Excel sheets containing information about different real estate projects.

## Requirements:

Design a scalable QnA System.
Implement as an API: Create a working API to showcase the system’s feasibility.
Functionality Testing: Test the code and report your observations on bugs, analysis, fixes, and performance.

# Features

* **PDF Processing**: Extracts and processes text from PDFs, including OCR for images within PDFs.
* **DOCX Processing**: Extracts text from DOCX files, including the content of any links embedded in the documents.
* **Image Processing**: Uses OCR to extract text from image files.
* **Excel Processing**: Processes Excel sheets and converts them into a text format suitable for querying.
* **FastAPI Integration**: Exposes the QnA system as a web API, allowing queries across different data types.

# Project Structure
* **pdf_bot.py**: Handles PDF files, including text and image processing.
* **docx_bot.py**: Handles DOCX files, including text extraction and link processing.
* **image_bot.py**: Handles image files, extracting text using OCR.
* **excel_bot.py**: Handles Excel files, processing each sheet into a text format.
* **main.py**: The FastAPI application that integrates all the bots and exposes them via different API routes.

# Prerequisites
* Python 3.10
* Poetry: Python dependency management tool.

# Installation

## Clone the Repository

     git clone git@github.com:PandeyRahul/PropLens-Assignment.git
     cd Proplens AI

## Set Up the Environment

**Install Poetry** (if not already installed):

     curl -sSL https://install.python-poetry.org | python3 -

Or follow the instructions on the [Poetry official website](https://python-poetry.org/docs/).

## Install Dependencies:

Ensure you're in the project directory, then run:

     poetry install
This will create a virtual environment and install all the necessary dependencies specified in the pyproject.toml file.

## Activate the Environment

To activate the Poetry environment, use:

     poetry shell

## Running the API

After setting up the environment and installing the dependencies, you can run the FastAPI server.


     uvicorn main:app --reload

This command will start the server on http://0.0.0.0:8000.

## API Endpoints

* **POST /query_pdf/**: Queries all PDF files located in the data directory.
* **POST /query_docx/**: Queries all DOCX files located in the data directory.
* **POST /query_image/**: Queries all image files (JPEG, JPG, PNG) located in the data/ directory.
* **POST /query_excel/**: Queries all Excel files located in the data directory.

## Example Usage

You can use tools like curl or Postman to interact with the API. Here are some examples using curl:

### Query PDFs:


     curl -X POST "http://localhost:8000/query_pdf/" -d "query=What is the lead registration policy?"

### Query DOCX files:

     curl -X POST "http://localhost:8000/query_docx/" -d "query=What are the sales policies?"

### Query Images:

     curl -X POST "http://localhost:8000/query_image/" -d "query=What does the floor plan look like?"

### Query Excel files:

     curl -X POST "http://localhost:8000/query_excel/" -d "query=Show me the inventory details.


## Acknowledgments

* **LangChain:** For the awesome libraries and tools that made this project possible.
* **Poetry:** For simplifying dependency management and project setup.
* **FastAPI:** For making API development fast and fun.