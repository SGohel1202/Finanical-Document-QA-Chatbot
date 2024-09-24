import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from streamlit_mic_recorder import speech_to_text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="Financial Document Insight retriever", layout="wide")

st.markdown("""
## Financial Document Insight retriever : 
###### Get instant insights from your Financial Documents. This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works
Follow these simple steps to interact with the chatbot:
1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.
2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

api_key = os.getenv("GOOGLE_API_KEY")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store for document retrieval
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the QA chain for question-answering
def get_conversational_chain():
    prompt_template = """
    You are an assistant for question-answering tasks for Retrieval Augmented Generation system for financial reports such as 10Q and 10K.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use two sentences maximum and keep the answer concise.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to retrieve and respond to user questions
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main app logic
def main():
    st.header("Your very own AI chatbot💁")

    # Check if there's already processed data in session
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "qa_history" not in st.session_state:  # Initialize question-answer history
        st.session_state.qa_history = []

    # Speech to Text Integration
    c1, c2 = st.columns(2)
    with c1:
        st.write("Speak or type a question:")
    with c2:
        speech = speech_to_text(language='en', use_container_width=True, key='STT')
    
    user_question = st.text_input("Ask a question from the PDF files", key="user_question")

    # If speech is detected, override the text input
    if speech:
        user_question = speech

    if user_question and api_key:
        if st.session_state.processed:
            with st.spinner("Fetching your answer..."):
                answer = user_input(user_question, api_key)
                st.write("Answer:", answer)
                
                # Store the question and answer in history
                st.session_state.qa_history.append({"question": user_question, "answer": answer})
        else:
            st.warning("Please upload and process PDF files first.")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Submit & Process") and api_key:
            with st.spinner("Processing..."):
                # Process and store document data in session
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.text_chunks = get_text_chunks(raw_text)
                get_vector_store(st.session_state.text_chunks, api_key)
                st.session_state.processed = True
                st.success("Documents processed successfully!")

    # Option to show question and answer history
    if st.checkbox("Show question and answer history"):
        if st.session_state.qa_history:
            for entry in st.session_state.qa_history:
                st.write(f"**Question**: {entry['question']}")
                st.write(f"**Answer**: {entry['answer']}")
        else:
            st.write("No history available.")

if __name__ == "__main__":
    main()
