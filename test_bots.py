import pytest
import os
from pdf_bot import handle_pdf_query
from excel_bot import handle_excel_query
from docx_bot import handle_docx_query
from image_bot import handle_image_query


def get_file_paths(extension: str or tuple):
    """Helper function to get file paths with a specific extension from the 'data' directory."""
    file_paths = []
    for dirpath, dirnames, filenames in os.walk("data"):
        for filename in filenames:
            if filename.endswith(extension):
                file_paths.append(os.path.join(os.path.abspath(dirpath), filename))
    return file_paths


def test_pdf_processing():
    query = "What are the property tax rates for owner-occupied residential properties?"
    pdf_paths = get_file_paths(".pdf")
    response = handle_pdf_query(query, pdf_paths)
    print(response)
    assert "The property tax rates for owner-occupied residential properties" in response


def test_excel_processing():
    query = "What is the total quantity of items across all inventory sheets?"
    excel_paths = get_file_paths(".xlsx")
    response = handle_excel_query(query, excel_paths)
    print(response)
    assert "total quantity of items" in response.lower()


def test_docx_processing():
    query = "What is the lead registration policy in the Sales SOP and policies document?"
    docx_paths = get_file_paths(".docx")
    response = handle_docx_query(query, docx_paths)
    print(response)
    assert "lead registration policy" in response.lower()


def test_image_processing():
    query = "What is the size and layout of the 1-bedroom + study unit in Tembusu Grand?"
    image_paths = get_file_paths((".jpg", ".jpeg", ".png"))
    response = handle_image_query(query, image_paths)
    print(response)
    assert "1-bedroom + study unit" in response.lower()
