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

def general_chatbot_response(query, bot_type="general"):
    """Handle general chat queries with different prompts for each bot type."""
    try:
        # Define prompt templates for each bot type
        if bot_type == "PDF Bot":
            prompt_template = f"Answer the following question based on the provided PDF context:\n\nQuestion: {query}\nAnswer:"

        elif bot_type == "ChatBot (Typing)" or bot_type == "ChatBot (Voice)":
            prompt_template = f"Respond as a friendly and helpful chatbot. Here's the question: {query}.Answer it in short sentence"

        elif bot_type == "Alexa-like Bot":
            prompt_template = f"You're an Alexa-like assistant. The user asked: {query}. Respond with the best possible answer in easy and small sentence if possible."

        else:
            prompt_template = f"Respond as a general chatbot. Here's the query: {query}. Answer in short sentence"

        # Send the prompt to Gemini
        response = chat_model.predict(prompt_template)
        return response
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

def record_audio(duration=5, samplerate=16000):
    """Record audio using sounddevice."""
    try:
        print("Recording...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        return audio_data
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

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
                answer = general_chatbot_response(question, bot_type="PDF Bot")
                st.write("Response:", answer)

    elif bot_type == "ChatBot (Typing)":
        st.subheader("ChatBot (Typing): General Chat")
        query = st.text_input("Ask your question here:")
        if query:
            with st.spinner("Thinking..."):
                answer = general_chatbot_response(query, bot_type="ChatBot (Typing)")
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
                        answer = general_chatbot_response(voice_query, bot_type="ChatBot (Voice)")
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
                        answer = general_chatbot_response(voice_query, bot_type="Alexa-like Bot")
                        st.write("Response:", answer)
                        text_to_speech(answer)

if __name__ == "__main__":
    main()
