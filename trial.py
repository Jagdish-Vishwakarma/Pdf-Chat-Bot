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
  # embeddings = OpenAIEmbeddings()

  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  print(embeddings)
  try:
      vector_store = FAISS.from_texts(chunks, embeddings)
  except Exception as e:
      print("Error in from_texts:", e)
      raise
  # vector_store = FAISS.from_texts(chunks, embeddings)
  vector_store.save_local("faiss_index")


def get_conversation():
  prompt_template = """
  Answer the question that is asked with as much detail as you can, given the context that has been provided. If you unable to come up with an answer based on the provided context,
  simply say "Answer cannot be provided based on the context that has been provided", instead of trying to forcibly provide an answer.\n\n
  Context: \n {context}?\n
  Question: \n {question}\n
  Answer:
  
  """

  model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

  prompt = PromptTemplate(template=prompt_template, input_variables = ["context", "question"])

  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  return chain 

def user_input(question):
  # embeddings = OpenAIEmbeddings()
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

  # Load embeddings from local file
  new_db = FAISS.load_local("faiss_index", embeddings)
  docs = new_db.similarity_search(question)

  chain = get_conversation()

  response = chain(
    {"input_documents": docs, "question": question}, return_only_outputs=True
  )

  st.write("Reply: ", response["output_text"])


# Main app portion of the project
def app():
  st.title("Public Company Financial Documents Analyzer")
  st.sidebar.title("Upload Documents")

  # Sidebar
  pdf_docs = st.sidebar.file_uploader("Upload your financial documents in PDF format, then click Analyze.", accept_multiple_files=True)

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
