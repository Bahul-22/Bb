import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize conversational model
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Helper Functions
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def query_pdf(question):
    """Query the processed PDF data."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings)
    docs = vector_store.similarity_search(question)

    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the context, then answer the question yourself on the level of class 9 NCERT, explaining how you answered.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response.get("output_text", "Sorry, I couldn't find an answer.")

# Main App
def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("PDF Bot: Ask Questions from Your PDFs")

    st.subheader("Upload PDF Files and Ask Questions")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("PDFs processed successfully! You can now ask questions.")
        else:
            st.error("Please upload at least one PDF file.")

    question = st.text_input("Ask a Question from the PDFs:")
    if question:
        with st.spinner("Thinking..."):
            answer = query_pdf(question)
            st.write("**Response:**", answer)

if __name__ == "__main__":
    main()
