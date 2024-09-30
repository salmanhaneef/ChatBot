import streamlit as st  
from dotenv import load_dotenv  
from PyPDF2 import PdfReader  
from langchain.text_splitter import CharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_huggingface import HuggingFaceEndpoint  
from langchain_community.vectorstores import FAISS  
from langchain.memory import ConversationBufferMemory  
from langchain.chains import ConversationalRetrievalChain  
from htmlTemplates import css, bot_template, user_template  
import os  
import requests  
import logging  
import time  

# Load environment variables  
load_dotenv()  
API_TOKEN = os.getenv("API_TOKEN")  

if not API_TOKEN:  
    st.error("API token not found or invalid in the .env file.")  
else:  
    st.success("API token loaded successfully.")  

def get_pdf_text(pdf_docs):  
    text = ""  
    for pdf in pdf_docs:  
        pdf_reader = PdfReader(pdf)  
        for page in pdf_reader.pages:  
            text += page.extract_text() or ""  
    return text  

def get_text_chunks(text):  
    text_splitter = CharacterTextSplitter(  
        separator="\n",  
        chunk_size=1000,  
        chunk_overlap=200,  
        length_function=len  
    )  
    chunks = text_splitter.split_text(text)  
    return chunks  

def get_vectorstore(text_chunks):  
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  
    return vectorstore  

logging.basicConfig(level=logging.INFO)  

def get_conversation_chain(vectorstore):  
    for attempt in range(3):  # Retry up to 3 times  
        try:  
            llm = HuggingFaceEndpoint(  
                repo_id="EleutherAI/gpt-neo-125M",  
                temperature=0.5,  
                max_length=1000,          
                max_new_tokens=520,       
                huggingfacehub_api_token=API_TOKEN,  
                request_timeout=60  # Increase timeout to 60 seconds  
            )  
            break  # Exit loop if successful  
        except requests.ReadTimeout:  
            logging.error("ReadTimeout error. Retrying...")  
            time.sleep(5)  # Wait before retrying  
        except Exception as e:  
            logging.error(f"Failed to initialize LLM: {str(e)}")  
            st.error(f"Failed to initialize LLM: {str(e)}")  
            return None  

    memory = ConversationBufferMemory(  
        memory_key='chat_history', return_messages=True  
    )  

    conversation_chain = ConversationalRetrievalChain.from_llm(  
        llm=llm,  
        retriever=vectorstore.as_retriever(),  
        memory=memory  
    )  
    
    return conversation_chain  

def handle_userinput(user_question):  
    if st.session_state.conversation:  
        response = st.session_state.conversation({'question': user_question})  
        st.session_state.chat_history = response.get('chat_history', [])  
        
        for i, message in enumerate(st.session_state.chat_history):  
            if i % 2 == 0:  
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  
            else:  
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  
    else:  
        st.warning("Please process your PDFs first.")  

def main():  
    st.write(css, unsafe_allow_html=True)  

    if "conversation" not in st.session_state:  
        st.session_state.conversation = None  
    if "chat_history" not in st.session_state:  
        st.session_state.chat_history = []  

    st.header("Chat with multiple PDFs :books:")  

    user_question = st.text_input("Ask a Question about your documents:")  
    if user_question:  
        handle_userinput(user_question)  

    st.subheader("Your documents")  
    pdf_docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)  

    if st.button("Process"):  
        if pdf_docs:  
            with st.spinner("Processing"):  
                raw_text = get_pdf_text(pdf_docs)  
                text_chunks = get_text_chunks(raw_text)  
                vectorstore = get_vectorstore(text_chunks)  
                st.success("Processing completed!")  

                st.write(f"Vectorstore created with {len(text_chunks)} chunks.")  
                
                st.session_state.conversation = get_conversation_chain(vectorstore)  
                if not st.session_state.conversation:  
                    st.error("Failed to create conversation chain.")  
        else:  
            st.warning("Please upload at least one PDF file.")  

if __name__ == '__main__':  
    main()