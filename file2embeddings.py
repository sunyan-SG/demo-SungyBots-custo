from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    #embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


st.set_page_config(page_title="file uploader",
                   layout="wide",
                   page_icon=":books::parrot:")

st.title("File Uploader for Sungy Customers")
st.session_state.pdf_docs = st.file_uploader(
    "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)
if st.button("Process"):
    # with st.spinner("Processing"):
    # get pdf text
    raw_text = get_pdf_text(st.session_state.pdf_docs)
    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    # create vector store
    vectorDB = get_vectorstore(text_chunks)

    # Save faiss vector to disk or store in a database/cloud storage
    file = st.session_state.pdf_docs[0]
    print(file)
    name = 'vectorDB_'+file.name
    vectorDB.save_local(name)

    st.success('PDF uploaded and embeddings created successfully!', icon="✅")