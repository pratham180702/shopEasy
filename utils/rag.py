from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
import time
import os
import logging
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from pydantic import BaseModel

load_dotenv()

log_path = os.getenv("LOG_PATH", "/var/log/main.log")

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

def configure_logs(log_path):
    global _logger

    try:
        if os.path.isdir(log_path):
            raise IsADirectoryError(f"{log_path} is a directory")

        final_log_path = log_path
        file_handler = logging.FileHandler(final_log_path)

    except Exception as e:
        final_log_path = os.path.join(os.getcwd(), "main.log")

        file_handler = logging.FileHandler(final_log_path)

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    _logger = logging.getLogger("rag_app")
    _logger.setLevel(logging.INFO)
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    _logger.propagate = False

    for logger_name in ["uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False

    _logger.info(f"---------RAG LOGGER CONFIGURED AT {final_log_path}----------")


configure_logs(log_path)

INDEX_NAME = "customer-support-agent-master"

pc = None
retriever = None
embeddings = None
llm = None
chain = None

def get_pinecone_client():
    global pc
    if pc is None:
        pc = Pinecone()
    return pc


#first step is to load the pfg
def load_pdf(file_path: str):
    if not file_path:
        return False
    loader = PyPDFLoader(file_path=file_path)

    docs = loader.load()

    return docs

#second step is to split the doc objects in the chunks
def split_docs_in_chunks(docs: Document) -> Document:

    if not docs:
        return False
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(docs)

    return chunks


def get_embedding_function():
    global embeddings

    if embeddings is not None:
        return embeddings

    try:
        _logger.info("Loading embedding model only once...")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings

    except Exception as e:
        _logger.error("There was some error in getting the embeddings")
        _logger.error(e)



def check_index_present_or_not() -> bool:
    pc = get_pinecone_client()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    return INDEX_NAME in existing_indexes

def create_and_embed_index():
    global retriever

    file_path = os.getenv("PDF_PATH")
    docs = split_docs_in_chunks(load_pdf(file_path))

    embeddings = get_embedding_function()

    pc = get_pinecone_client()
    vector_store = None

    if not pc.has_index(INDEX_NAME):

        _logger.info("INDEX NOT FOUND!!! Creating new index!")

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        _logger.info("Waiting for Pinecone index to become ready...")

        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)

        _logger.info("INDEX READY!")

        _logger.info("Embedding the docs chunks and storing into the index")

        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        _logger.info("CHUNKING AND EMBEDDING into the index done!")

    else:
        _logger.info("INDEX FOUND! Using existing index.")

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )

    retriever = vector_store.as_retriever()

    return retriever

def get_prompt_runnable() -> PromptTemplate:
    return PromptTemplate(
        template='''
                ### SYSTEM PROMPT

                You are a Customer Support Representative for ShopEasy, an e-commerce company. Your primary responsibility is to assist customers by providing accurate, helpful, and policy-compliant responses based on the information available to you.

                ## Guidelines

                ### 1. Maintain Professionalism

                - Communicate in a courteous, professional, and customer-friendly manner at all times.
                - Focus on resolving customer concerns clearly and efficiently.

                ### 2. Provide a Human-Like Support Experience

                - Respond naturally and conversationally.
                - Do not mention that you are an AI, chatbot, language model, or automated system unless explicitly asked.

                ### 3. Represent the Company

                - Greet customers warmly on behalf of ShopEasy.
                - Ensure all communication reflects the company's commitment to customer satisfaction.

                ### 4. Use Available Information Responsibly

                - Prioritize answering questions using the provided context and company knowledge.
                - If the required information is not available in the provided context, you may use external sources or tools to find the answer.
                - When external information is used:
                - Clearly inform the customer that the response is based on information obtained from an external source rather than ShopEasy's internal knowledge base.
                - Provide the source or reference used to obtain the information.
                - Avoid presenting externally sourced information as official ShopEasy policy unless explicitly confirmed.

                ### 5. Avoid Technical Terminology

                - Use simple, customer-friendly language.
                - Avoid technical, internal, or implementation-specific terms that may confuse customers.
                - Explain concepts in a clear and easy-to-understand manner.

                ### 6. Accuracy and Transparency

                - Do not make up information.
                - If you are uncertain about an answer, clearly communicate the limitation and provide the best available guidance.
                - When appropriate, recommend contacting a human support representative for further assistance.

                ---

                ## Customer Query

                {user_query}

                ## Available Context

                {context}
                ''',
                input_variables=['user_query', 'context']
    )


def get_llm() -> ChatGroq:
    global llm

    if llm is None:
        llm = ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0.3
        )

    return llm


def format_docs(docs):
    """
    Formats a list of retrieved Document objects into a single readable string.
    """
    formatted_docs =  "\n\n".join(doc.page_content for doc in docs)

    _logger.info("FORMATTED DOCS: ")
    _logger.info(formatted_docs)

    return formatted_docs

def get_retriever():
    global retriever

    if retriever is None:
        retriever = create_and_embed_index()

    return retriever


@tool
def web_search(query: str) -> str:
    """
    Search the web when internal documentation
    is insufficient.
    """

    _logger.info("INSIDE WEB SEARCH!!!!!!!!!!!!!!!!!!!!!")

    try:
        search = DuckDuckGoSearchResults(
            max_results=5
        )

        result = search.invoke(query)

        _logger.info("QUERY SEARCH DONE!!!!!!!!!!!!!")
        _logger.info(result)

        return str(result)

    except Exception as e:
        _logger.info(e)
        _logger.info("------INSIDE EXCEPTION")
        return f"Search failed: {str(e)}"


