import os 
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import sounddevice as sd
import numpy as np
import pydub.playback
from pydub import AudioSegment  
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Set ffmpeg path explicitly if necessary
AudioSegment.ffmpeg = "E:/Gemini/ffmpeg/bin/ffmpeg.exe"

# Load environment variables
load_dotenv()

# Initialize conversational model
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Helper Functions
def text_to_speech(text):
    """Convert text to speech and play it instantly."""
    if not text.strip():
        return
    tts = gTTS(text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        audio = AudioSegment.from_file(temp_audio.name)
        fast_audio = audio.speedup(playback_speed=1.3)
        pydub.playback.play(fast_audio)

def speech_to_text():
    """Convert speech to text using sounddevice for audio capture."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand. Please try again."
        except sr.RequestError:
            return "Error with the speech recognition service."

def detect_wakeword():
    """Continuously listen for the wake word 'UP' using sounddevice."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                query = recognizer.recognize_google(audio)
                if "UP" in query.upper():
                    return True
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                st.error("Microphone or recognition service error.")
                break
    return False

def get_pdf_text(pdf_docs=None):
    """Extract text from uploaded or default PDFs."""
    text = ""
    if not pdf_docs:
        default_pdf = "default.pdf"  # Path to a default PDF
        pdf_docs = [open(default_pdf, "rb")]
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

def general_chatbot_response(query):
    """Handle general chat queries."""
    try:
        response = chat_model.predict(query)
        return response
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

# Main App
def main():
    st.set_page_config(page_title="Bahul's AI Assistant", layout="wide")
    st.title("Bahul's Bot: Multipurpose AI Assistant")

    bot_type = st.sidebar.radio(
        "Choose Bot",
        ["PDF Bot", "ChatBot (Typing)", "ChatBot (Voice)", "Alexa-like Bot"],
        index=0
    )

    if bot_type == "PDF Bot":
        st.subheader("PDF Bot: Ask Questions from Uploaded PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("PDFs processed successfully! You can now ask questions.")

        question = st.text_input("Ask a Question from the PDFs:")
        if question:
            with st.spinner("Thinking..."):
                answer = query_pdf(question)
                st.write("Response:", answer)

    elif bot_type == "ChatBot (Typing)":
        st.subheader("ChatBot (Typing): General Chat")
        query = st.text_input("Ask your question here:")
        if query:
            with st.spinner("Thinking..."):
                answer = general_chatbot_response(query)
                st.write("Response:", answer)

    elif bot_type == "ChatBot (Voice)":
        st.subheader("ChatBot (Voice): Talk with Bahul's Bot")
        if st.button("Start Voice Call"):
            st.info("Speak your query. End call when done.")
            end_call = False
            while not end_call:
                voice_query = speech_to_text()
                if voice_query.lower() == "end call":
                    end_call = True
                    st.success("Call ended.")
                else:
                    with st.spinner("Thinking..."):
                        answer = general_chatbot_response(voice_query)
                        st.write("Response:", answer)
                        text_to_speech(answer)

    elif bot_type == "Alexa-like Bot":
        st.subheader("Alexa-like Bot: Always Listening")
        while True:
            if detect_wakeword():
                st.success("Wake word detected: 'UP'")
                st.info("Listening for your query...")
                voice_query = speech_to_text()
                if voice_query:
                    with st.spinner("Thinking..."):
                        answer = general_chatbot_response(voice_query)
                        st.write("Response:", answer)
                        text_to_speech(answer)

if __name__ == "__main__":
    main()
