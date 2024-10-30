import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

## Set the Google API key directly in the code
GOOGLE_API_KEY = "AIzaSyAsnH_vjbRiBG3C8PD9-40DTK4NCBfDZTg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Set up Google Generative AI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "Answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    
    normalized_question = user_question.lower()
    
    # Check if the question is a greeting
    if any(greet in normalized_question for greet in greetings):
        return "Hello! How can I assist you today?"

    # Normalization for specific terms
    keyword_mapping = {
        "hod": "head",
        "cse": "computer science and engineering",
        "ece": "electrical and communication engineering",
        "eee": "electrical and electronics engineering",
        "ai": "artificial intelligence",
        "ds": "data science",
    }

    for key, value in keyword_mapping.items():
        normalized_question = normalized_question.replace(key, value)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(normalized_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": normalized_question}, return_only_outputs=True)

    return response["output_text"]

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="VRSEC College Chatbot", layout="wide")
    
    # Title of the Streamlit app
    st.title("🎓 VRSEC College Chatbot")

    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Specify the PDF file paths
    pdf_file_paths = [
        "VRSEC.pdf",  # Replace with your actual PDF paths
        # Add more paths as needed
    ]

    # Process the specified PDF files
    raw_text = get_pdf_text(pdf_file_paths)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Input box for user query
    if prompt := st.chat_input("Ask me anything about VRSEC (e.g., 'What programs are offered?', 'Who is the principal?')"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in the chat
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate content based on user input
        response = user_input(prompt)
        
        # Display assistant response in the chat
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()
