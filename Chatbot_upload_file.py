import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract all text from uploaded PDFs
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)

        # Iterate through all the pages in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

# Split text into chunks
def generate_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200  # Reduced overlap for efficiency
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Convert chunks into vectors (FAISS index in-memory)
def chunks_to_vectors(chunks):
    # Initialize the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize an in-memory FAISS index
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# Get conversation chain
def get_conversation():
    prompt_template = """
    Answer the question that is asked with as much detail as you can, given the context that has been provided. 
    If you are unable to provide an answer based on the provided context, simply say 
    'Answer cannot be provided based on the context that has been provided', instead of forcing an answer.
    All the files uploaded are directly or indirectly related to Additive Manufacturing and 3D Printing industry and companies.
    
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain 

# Handle user input
def user_input(question, vector_store):
    # Perform similarity search on the FAISS index
    docs = vector_store.similarity_search(question)
    
    # Get the conversation chain
    chain = get_conversation()

    # Get the response from the chain
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

# Main app
def app():
    st.title("ASTM Documents Chatbot")
    st.sidebar.title("Upload Documents")

    # Sidebar: Upload PDF documents
    pdf_docs = st.sidebar.file_uploader("Upload your documents in PDF format, then click on Chat Now.", accept_multiple_files=True)

    analyze_triggered = st.sidebar.button("Chat Now")

    # This will store the FAISS index globally for the session
    if analyze_triggered:
        with st.spinner("Configuring... ⏳"):
            # Get the extracted text from the PDFs
            raw_text = get_pdf_text(pdf_docs)

            # Check if the extracted text is None or empty
            if raw_text is None or raw_text.strip() == "":
                st.error("No text extracted from the PDF. Please check the file.")
                return

            # Split text into chunks and create FAISS index
            chunks = generate_chunks(raw_text)
            vector_store = chunks_to_vectors(chunks)
            st.session_state.vector_store = vector_store  # Store FAISS index in session state
            st.success("Documents processed. You can now ask questions.")

    # User Input 
    user_question = st.text_input("Ask a question based on the documents that were uploaded")

    if user_question:
        if 'vector_store' in st.session_state:
            # Use the vector store stored in the session state
            user_input(user_question, st.session_state.vector_store)
        else:
            st.warning("Please upload and process the documents first.")

if __name__ == "__main__":
    app()
