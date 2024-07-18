#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import fitz  # PyMuPDF
# import pandas as pd


# In[2]:


# # Function to extract text from a specific rectangular area of a page
# def extract_text_from_rect(page, rect):
#     # Extract text directly from the rectangle area of the page
#     clip = page.get_text("blocks", clip=rect)
#     text = ""
#     for block in clip:
#         text += block[4] + "\n"
#     return text

# # Function to split a page vertically and extract text from both parts
# def split_and_extract_text(page, remove_bottom=0):
#     rect = page.rect
#     mid_x = rect.width / 2
#     page_height = rect.height

#     # Adjust bottom of the rectangle to remove a specified portion
#     top_y = rect.tl.y
#     bottom_y = rect.br.y - remove_bottom
#     adjusted_rect = fitz.Rect(rect.tl, fitz.Point(rect.br.x, bottom_y))

#     # Define left and right rectangles
#     left_rect = fitz.Rect(rect.tl, fitz.Point(mid_x, bottom_y))
#     right_rect = fitz.Rect(fitz.Point(mid_x, top_y), fitz.Point(rect.br.x, bottom_y))

#     # Extract text from left part
#     left_text = extract_text_from_rect(page, left_rect)

#     # Extract text from right part
#     right_text = extract_text_from_rect(page, right_rect)

#     return left_text, right_text

# # Path to the original PDF
# pdf_path = "ICAM_2023.pdf"

# # Open the PDF
# doc = fitz.open(pdf_path)

# # Extract and arrange text
# arranged_text = ""

# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
    
#     # Remove 100 pixels (adjust this value as needed) from the bottom before splitting
#     left_text, right_text = split_and_extract_text(page, remove_bottom=50)
    
#     arranged_text += left_text + "\n" + right_text + "\n"

# # Print the arranged text
# print(arranged_text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# In[20]:


import os
import streamlit as st
import google.generativeai as genai

from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv


# In[35]:


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Extract all of the texts from the PDF files 
def get_pdf_text(pdfs):
  text = ""
  for pdf in pdfs:
    pdf_reader = PdfReader(pdf)

    # iterate through all the pages in the pdf
    for page in pdf_reader.pages:
      text += page.extract_text()
    
    return text 

# Split text into chunks 
def generate_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=1000
  )

  chunks = text_splitter.split_text(text)

  return chunks 

# Convert Chunks into Vectors
def chunks_to_vectors(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        vector_store.save_local("faiss_index")  # Save the updated index
    except Exception as e:
        print("Error in loading FAISS index:", e)
        raise
  # vector_store = FAISS.from_texts(chunks, embeddings)
   #vector_store.save_local("faiss_index")


def get_conversation():
    prompt_template = """
    Answer the question that is asked with as much detail as you can, given the context that has been provided. If you unable to come up with an answer based on the provided context,
    simply say "Answer cannot be provided based on the context that has been provided", instead of trying to forcibly provide an answer.\n\n
    Context: \n {context}?\n
    Question: \n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain 

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    chain = get_conversation()

    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])


# Main app portion of the project
# Main app portion of the project
def app():
    st.title("Documents Analyzer")
    st.sidebar.title("Upload Documents")

    # Sidebar
    pdf_docs = st.sidebar.file_uploader("Upload your documents in PDF format, then click Analyze.", accept_multiple_files=True)

    analyze_triggered = st.sidebar.button("Analyze")

    if analyze_triggered:
        with st.spinner("Analyzing... ⏳"):
            raw_text = get_pdf_text(pdf_docs)
            chunks = generate_chunks(raw_text)
            chunks_to_vectors(chunks)
            st.success("Done")

    # User Input 
    user_question = st.text_input("Ask a question based on the shareholder letters that were uploaded")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    app()





# In[ ]:





# In[ ]:





# In[ ]:




