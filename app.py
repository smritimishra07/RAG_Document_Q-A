import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings 
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize LLM and Embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings and store them in session state
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.write("Creating vector embeddings...")
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.write(f"Loaded {len(st.session_state.docs)} documents.")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.write(f"Split documents into {len(st.session_state.final_documents)} chunks.")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector database created.")

# Streamlit App UI
st.title("RAG Document Q&A With Groq And Lama3")

# User input for the query
user_prompt = st.text_input("Enter your query from the research paper")

# Button to trigger vector embedding creation
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

# Check if the user has entered a prompt
if user_prompt:
    # Ensure vector embeddings are created before proceeding
    if "vectors" not in st.session_state:
        st.warning("Please create the vector embeddings first by clicking the 'Document Embedding' button.")
    else:
        # Create document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process the query and get the response
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start}")

        # Display the answer to the user
        st.write(response['answer'])

        # Show document similarity search results in an expander
        with st.expander("Document similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content)
                st.write('------------------------')