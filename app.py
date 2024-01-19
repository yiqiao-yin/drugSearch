import openai
import pandas as pd
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from openai import OpenAI

from helpers.foundation_models import *

# Set the OpenAI API key. This key is retrieved from a Streamlit's secrets management system,
# which securely stores and accesses sensitive information like API keys.
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Initialize an OpenAI client. This client will be used to interact with OpenAI's API.
openai_client = openai.OpenAI()


# Define a list of file names. These files contain data that will likely be used for processing.
# The path suggests that these are text files related to different phases of a study or project
# involving 'Exenatide', which might be a drug or a medical treatment.
file_names = [
    "data/txt/Exenatide Phase 1.1_0.txt",
    "data/txt/Exenatide Phase 1.1_1.txt",
    "data/txt/Exenatide Phase 1.1_2.txt",
    "data/txt/Exenatide Phase 1.2_0.txt",
    "data/txt/Exenatide Phase 1.2_1.txt",
    "data/txt/Exenatide Phase 1.2_2.txt",
    "data/txt/Exenatide Phase 1.1_3.txt",
    "data/txt/Exenatide Phase 1.2_3.txt",
]

# Create a Chroma database by calling the `build_chromadb` function with the list of file names.
# The `chromadb` variable now holds the Chroma database instance.
db = build_chromadb(list_files=file_names)

# Set the title of the Streamlit web page to "Echo Bot".
st.title("Echo Bot")

# Initialize the chat history in the Streamlit session state if it doesn't already exist.
# The session state persists across reruns of the app, enabling continuity in the chat.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display each message in the chat history.
# This loop goes through all the messages saved in the session state and displays them on the web page.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for the user to enter their message.
# 'st.chat_input' creates a chat-style input box with the prompt "What is up?"
if prompt := st.chat_input("What is up?"):
    # Display the user's message in a chat message container labeled as "user".
    st.chat_message("user").markdown(prompt)
    # Add the user's message to the chat history in the session state.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create a response string that echoes the user's input.
    response = f"Echo: {prompt}"
    query = f"User origin question:{prompt}"
    docs = db.similarity_search(query)
    top_n = 2
    retrieved_documents = " ".join([docs[i].page_content for i in range(top_n)])
    response = rag(query=query, retrieved_documents=retrieved_documents)
    references = pd.DataFrame(
        [[docs[i].metadata["source"], docs[i].page_content] for i in range(top_n)]
    )

    # Display the assistant's response in a chat message container labeled as "assistant".
    with st.chat_message("assistant"):
        st.markdown(response)
        st.markdown(references)
    # Add the assistant's response to the chat history in the session state.
    st.session_state.messages.append({"role": "assistant", "content": response})
