from typing import List, Tuple
import streamlit as st
import PyPDF2


def read_and_textify(
    files: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Reads PDF files and extracts text from each page.

    This function iterates over a list of uploaded PDF files, extracts text from each page,
    and compiles a list of texts and corresponding source information.

    Args:
    files (List[st.uploaded_file_manager.UploadedFile]): A list of uploaded PDF files.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists:
        1. A list of strings, where each string is the text extracted from a PDF page.
        2. A list of strings indicating the source of each text (file name and page number).
    """

    # Initialize lists to store extracted texts and their sources
    text_list = []  # List to store extracted text
    sources_list = []  # List to store source information

    # Iterate over each file
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)  # Create a PDF reader object
        # Iterate over each page in the PDF
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]  # Get the page object
            text = pageObj.extract_text()  # Extract text from the page
            pageObj.clear()  # Clear the page object (optional, for memory management)
            text_list.append(text)  # Add extracted text to the list
            # Create a source identifier and add it to the list
            sources_list.append(file.name + "_page_" + str(i))

    # Return the lists of texts and sources
    return [text_list, sources_list]
