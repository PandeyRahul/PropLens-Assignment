import os
from fastapi import FastAPI, HTTPException
import uvicorn

# Import the bot functions from your existing implementations
from pdf_bot import handle_pdf_query
from docx_bot import handle_docx_query
from image_bot import handle_image_query
from excel_bot import handle_excel_query

app = FastAPI()


def get_file_paths(extension: str):
    """Helper function to get file paths with a specific extension from the 'data' directory."""
    file_paths = []
    for dirpath, dirnames, filenames in os.walk("data"):
        for filename in filenames:
            if filename.endswith(extension):
                file_paths.append(os.path.join(os.path.abspath(dirpath), filename))
    return file_paths


# Route for querying PDF files
@app.post("/query_pdf/")
async def query_pdf(query: str):
    pdf_paths = get_file_paths(".pdf")

    if not pdf_paths:
        raise HTTPException(status_code=404, detail="No PDF files found in the data directory")

    response = handle_pdf_query(query, pdf_paths)
    return {"result": response}


# Route for querying DOCX files
@app.post("/query_docx/")
async def query_docx(query: str):
    docx_paths = get_file_paths(".docx")

    if not docx_paths:
        raise HTTPException(status_code=404, detail="No DOCX files found in the data directory")

    response = handle_docx_query(query, docx_paths)
    return {"result": response}


# Route for querying Image files
@app.post("/query_image/")
async def query_image(query: str):
    image_extensions = [".jpeg", ".jpg", ".png"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(get_file_paths(ext))

    if not image_paths:
        raise HTTPException(status_code=404, detail="No image files found in the data directory")

    response = handle_image_query(query, image_paths)
    return {"result": response}


# Route for querying Excel files
@app.post("/query_excel/")
async def query_excel(query: str):
    excel_paths = get_file_paths(".xlsx")

    if not excel_paths:
        raise HTTPException(status_code=404, detail="No Excel files found in the data directory")

    response = handle_excel_query(query, excel_paths)
    return {"result": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
