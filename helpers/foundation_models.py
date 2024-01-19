from typing import List

import openai
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from openai import OpenAI

# Set the OpenAI API key. This key is retrieved from a Streamlit's secrets management system,
# which securely stores and accesses sensitive information like API keys.
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Initialize an OpenAI client. This client will be used to interact with OpenAI's API.
openai_client = openai.OpenAI()


def build_chromadb(list_files: List[str]) -> Chroma:
    """
    Builds a Chroma database from a list of text files.

    The function reads the content of each file, splits the text into chunks,
    applies embeddings, and loads the data into a Chroma database.

    Args:
    list_files (List[str]): A list of file names to be loaded.

    Returns:
    Chroma: An instance of the Chroma database populated with the processed documents.
    """

    # Initialize an empty list to hold all documents
    file_names = list_files  # List of file names provided as input

    all_documents = []  # List to store all loaded documents

    # Iterate over each file and load its contents
    for file_name in file_names:
        loader = TextLoader(file_name)  # Initialize a text loader for the file
        documents = loader.load()  # Load the contents of the file
        all_documents.extend(documents)  # Add the loaded documents to the list

    # Split the loaded documents into chunks
    # Assuming a character-based text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(all_documents)  # Split documents into chunks

    # Create the open-source embedding function
    # Using a pre-trained sentence transformer model for embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the documents into Chroma
    db = Chroma.from_documents(
        docs, embedding_function
    )  # Create Chroma DB from documents

    return db


def rag(
    query: str, retrieved_documents: List[str], model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generates a response to a query using a Retrieve-And-Generate (RAG) approach with OpenAI's GPT model.

    This function takes a user query and a list of retrieved documents relevant to the query,
    then uses the specified GPT model to generate a response based on this information.

    Args:
    query (str): The user's query or question.
    retrieved_documents (List[str]): A list of documents (strings) retrieved as relevant to the query.
    model (str, optional): The name of the GPT model to use. Defaults to "gpt-3.5-turbo".

    Returns:
    str: The generated response from the model.
    """

    # Join the retrieved documents into a single string, separated by double newlines
    information = "\n\n".join(retrieved_documents)

    # Prepare the messages to be sent to the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful expert assistant. "
                "You will be shown the user's question, and the relevant information from the instructions. "
                "Answer the user's question using only this information."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}. \n Information: {information}",
        },
    ]

    # Generate a response using the OpenAI GPT model
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract and return the content of the response
    content = response.choices[0].message.content
    return content
