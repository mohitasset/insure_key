from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from groq import Groq
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.header("Welcome to Assetplus Insurance Dictionary")

messages = []

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def load_pdf():
    loader = PyPDFDirectoryLoader(path="docs")
    return loader.load()

def split_docs(docs):
    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
        add_start_index = True
    )

    return text_splitters.split_documents(docs)


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(
        collection_name="insure_key",
        embedding_function=embeddings,
        persist_directory="./chromadb"
    )
    return vector_store

def create_store_embeddings(chunks):
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store = get_vector_store()
    vector_store.add_documents(documents=chunks, ids=uuids)


def store_data():
    docs = load_pdf()
    chunks = split_docs(docs)
    create_store_embeddings(chunks)


def search_vector_db(query):
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=5)
    return results

def generate_response(query, context):
    prompt = f"""
    You have all the knowledge related to the Insure key in Assetplus. You have the ability to look at the context and give the answer accordingly. You always tend to give the right answer if the answer is not given in the context then reply with I don't know.
    """
    messages.append({"role" : "user", "content" : f"""
    Query : {query}
    Context : {context}
    """})

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content
        

user_question = st.text_input(label="",placeholder="Ask your query....")

if user_question:
    context = search_vector_db(user_question)
    res = generate_response(user_question, context)
    st.write(res)