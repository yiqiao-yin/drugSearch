import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

from helpers.foundation_models import *


st.set_page_config(layout="centered", page_title="Drug Search🤖💊")
st.header("Drug Search🤖💊")
st.write("---")

# file uploader
with st.sidebar:
    with st.expander("Instruction Manual 📖"):
        st.markdown(
            """
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

    uploaded_files = st.file_uploader(
        "Upload documents", accept_multiple_files=True, type=["txt", "pdf"]
    )
    st.write("---")


if uploaded_files is None:
    st.info(f"""Upload files to analyse""")
elif uploaded_files:
    st.write(str(len(uploaded_files)) + " document(s) loaded..")

    textify_output = read_and_textify(uploaded_files)

    documents = textify_output[0]
    sources = textify_output[1]

    # extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # vstore with metadata. Here we will store page numbers.
    vStore = Chroma.from_texts(
        documents, embeddings, metadatas=[{"source": s} for s in sources]
    )

    # deciding model
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {"k": 2}

    # initiate model
    llm = OpenAI(
        model_name=model_name,
        openai_api_key=st.secrets["openai_api_key"],
        streaming=True,
    )

    model = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")

    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = model({"question": user_q}, return_only_outputs=True)
                st.subheader("Your response:")
                st.write(result["answer"])
                st.subheader("Source pages:")
                st.write(result["sources"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(
                "Oops, the GPT response resulted in an error :( Please try again with a different question."
            )
