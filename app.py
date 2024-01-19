import langchain
import PyPDF2
import streamlit as st
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from helpers.foundation_models import *

st.set_page_config(layout="centered", page_title="Drug Search🤖💊")
st.header("Drug Search🤖💊")
st.write("---")

# Streamlit sidebar setup for user interface
with st.sidebar:
    # Create an expandable instruction manual section in the sidebar
    with st.expander("Instruction Manual 📖"):
        # Display the instruction manual for the Exenatide Chatbot in a formatted markdown
        st.markdown(
            """
            # Exenatide Chatbot User Manual 🤖💊
            
            Welcome to the Exenatide Chatbot, your interactive assistant for information on the drug "Exenatide". This chatbot offers quick and accurate responses to your queries. Follow these steps to interact with the chatbot:

            ## Getting Started 🚀
            1. **Access the Chatbot**: Launch the Exenatide Chatbot on your device.
            2. **Start Chatting**: Type your Exenatide-related questions in the chat window. Questions can range from dosage to side effects.
            3. **Send Your Question**: Submit your query by clicking 'Send' or pressing 'Enter'.

            ## Chatting with Exenatide Chatbot 🤔💬
            - **Ask Anything**: Inquiries about drug composition, usage, storage, or safety are all welcome.
            - **Use Simple Language**: Clear and concise questions yield the best results.
            - **Wait for the Response**: The chatbot will promptly process and answer your query.
            - **Follow-Up Questions**: Feel free to ask additional or new questions anytime.

            ## Tips for a Better Experience ✨
            - **Be Specific**: Specific questions help in getting precise answers.
            - **Check for Typing Errors**: Correct spelling ensures better understanding by the chatbot.
            - **Emoji Use**: Emojis are welcome in your questions!
            - **Patience is Key**: Responses may take a moment as the chatbot processes your query.

            ## Support and Feedback 🤝
            - **Need Help?**: Contact our support team for any issues.
            - **Share Your Feedback**: Your input is valuable and helps us improve.

            ## The Team Behind the App 🧑‍💻👩‍💻
            - **Founders**: Learn about Peter Yin and Yiqiao Yin, the founders, on LinkedIn.

            Thank you for choosing the Exenatide Chatbot. We're here to provide all the information you need about Exenatide efficiently. Happy chatting! 🎉💬
            """
        )

    # File uploader widget allowing users to upload text and PDF documents
    uploaded_files = st.file_uploader(
        "Upload documents", accept_multiple_files=True, type=["txt", "pdf"]
    )
    # A separator line for visual clarity in the sidebar
    st.write("---")


# Check if any files have been uploaded
if uploaded_files is None:
    # Display a message prompting the user to upload files
    st.info("Upload files to analyze")
elif uploaded_files:
    # Inform the user how many documents have been loaded
    st.write(f"{len(uploaded_files)} document(s) loaded..")

    # Process the uploaded files to extract text and source information
    textify_output = read_and_textify(uploaded_files)

    # Separate the output into documents (text) and their corresponding sources
    documents, sources = textify_output

    # Initialize OpenAI embeddings with the API key from Streamlit secrets
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Create a Chroma Vector Store with metadata. This store will include page numbers as metadata.
    vStore = Chroma.from_texts(
        documents, embeddings, metadatas=[{"source": s} for s in sources]
    )

    # Define the model to use for the language model, here 'gpt-3.5-turbo'
    model_name = "gpt-3.5-turbo"
    # Alternative model: model_name = "gpt-4"

    # Set up a retriever using the Chroma Vector Store
    retriever = vStore.as_retriever()
    retriever.search_kwargs = {"k": 2}  # Set the number of documents to retrieve

    # Initialize the language model with the specified model name and API key
    llm = OpenAI(
        model_name=model_name,
        openai_api_key=st.secrets["openai_api_key"],
        streaming=True,
    )

    # Set up a retrieval-qa chain using the language model and retriever
    model = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    # User interface to ask questions
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")

    # Button to get the response from the model
    if st.button("Get Response"):
        try:
            # Display a spinner while the model is processing the question
            with st.spinner("Model is working on it..."):
                # Get the response from the model
                result = model({"question": user_q}, return_only_outputs=True)
                st.subheader("Your response:")
                st.write(result["answer"])  # Display the answer
                st.subheader("Source pages:")
                st.write(result["sources"])  # Display the source pages
        except Exception as e:
            # Handle exceptions by displaying an error message
            st.error(f"An error occurred: {e}")
            st.error(
                "Oops, the GPT response resulted in an error :( Please try again with a different question."
            )
