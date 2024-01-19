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
# Create a Chroma database by calling the `build_chromadb` function with the list of file names.
# The `chromadb` variable now holds the Chroma database instance.
@st.cache_resource(ttl="1h")
db = load_vector_db("vectorstore")
st.write(type(db))

query = f"User origin question: what is exenatide?"
st.write(query)
docs = db.similarity_search(query)
st.write(docs)

# Set the title of the Streamlit web page to "Drug Search🤖💊".
st.title("Drug Search🤖💊")


# Sidebars
with st.sidebar:
    with st.expander("Instruction Manual 📖"):
        st.markdown(
            r"""
            # Exenatide Chatbot User Manual 🤖💊

            Welcome to the Exenatide Chatbot, your go-to assistant for all information about the proprietary drug "Exenatide." This easy-to-use chatbot is designed to provide quick, reliable answers to your questions about Exenatide. Follow these simple steps to start chatting!

            ## Getting Started 🚀

            1. **Access the Chatbot**: Open the Exenatide Chatbot application on your preferred device.

            2. **Start Chatting**: Simply type your question about Exenatide into the chat window. It could be anything from dosage information to side effects or general inquiries about the drug.

            3. **Send Your Question**: Press the 'Send' button or hit 'Enter' to submit your question to the chatbot.

            ## Chatting with Exenatide Chatbot 🤔💬

            - **Ask Anything**: Whether it’s detailed drug composition, usage guidelines, storage instructions, or safety precautions, feel free to ask. Example: "What is the recommended dosage of Exenatide for adults?"

            - **Use Simple Language**: For best results, use clear and concise questions. The chatbot is designed to understand everyday language.

            - **Wait for the Response**: Once you submit your question, the chatbot will process it and provide an answer shortly.

            - **Follow-Up Questions**: You can ask follow-up questions or new questions at any time. Just type them into the chat window and send.

            ## Tips for a Better Experience ✨

            - **Be Specific**: The more specific your question, the more accurate the chatbot's response will be.

            - **Check for Typing Errors**: Ensure your question is free from typos to help the chatbot understand you better.

            - **Emoji Use**: Feel free to use emojis in your questions! The chatbot is emoji-friendly. 😊

            - **Patience is Key**: If the chatbot takes a moment to respond, don't worry. It's just processing the best possible answer for you!

            ## Support and Feedback 🤝

            - **Experiencing Issues?**: If you face any issues or have technical difficulties, please contact our support team.

            - **We Value Your Feedback**: After using the Exenatide Chatbot, we would love to hear your thoughts. Your feedback helps us improve!

            ## The Team Behind the App 🧑‍💻👩‍💻

            The Exenatide Chatbot is proudly founded by Peter Yin and Yiqiao Yin. Their dedication and expertise have been instrumental in bringing this innovative solution to life.

            - **Peter Yin**: Get to know more about Peter and his professional journey on [LinkedIn](https://www.linkedin.com/in/peter-yin-7914ba25/).

            - **Yiqiao Yin**: Discover more about Yiqiao's background and accomplishments on [LinkedIn](https://www.linkedin.com/in/yiqiaoyin/).

            Thank you for using the Exenatide Chatbot! We hope it helps you find all the information you need about Exenatide quickly and easily. Happy chatting! 🎉💬
            """
        )
clear_button = st.sidebar.button("Clear Conversation", key="clear")


# # Initialize the chat history in the Streamlit session state if it doesn't already exist.
# # The session state persists across reruns of the app, enabling continuity in the chat.
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Clear the chat history in the Streamlit session state if it gets long or messy.
# # The session state persists across reruns of the app, enabling continuity in the chat.
# if clear_button:
#     st.session_state.messages = []

# # Display each message in the chat history.
# # This loop goes through all the messages saved in the session state and displays them on the web page.
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Input field for the user to enter their message.
# # 'st.chat_input' creates a chat-style input box with the prompt "What is up?"
# if prompt := st.chat_input("What is up?"):
#     # Display the user's message in a chat message container labeled as "user".
#     st.chat_message("user").markdown(prompt)
#     # Add the user's message to the chat history in the session state.
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Create a response string that echoes the user's input.
#     response = f"Drug Search🤖💊: {prompt}"
#     query = f"User origin question:{prompt}"
#     docs = db.similarity_search(query)
#     retrieved_documents = " ".join([docs[i].page_content for i in range(len(docs))])
#     response = rag(query=query, retrieved_documents=retrieved_documents)

#     # Create reference table
#     references = pd.DataFrame(
#         [[docs[i].metadata["source"], docs[i].page_content] for i in range(len(docs))]
#     )
#     if references.shape[1] >= 2:
#         references.columns = ["Source", "Excerpt"]

#     # Display the assistant's response in a chat message container labeled as "assistant".
#     with st.chat_message("assistant"):
#         st.markdown(response)
#         st.write("#### Reference")
#         st.write(references)
#         st.write(docs)
#     # Add the assistant's response to the chat history in the session state.
#     st.session_state.messages.append({"role": "assistant", "content": response})
