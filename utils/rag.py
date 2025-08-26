import os
from pathlib import Path
from typing import List, Union
from utils.logger import logger
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def build_or_load_vector_store(
    documents=None,
    persist_directory: str = "vector_store",
    collection_name: str = "collection",
    force_rebuild: bool = False,
) -> Union[Chroma, None]:
    """
    Build new or load existing Chroma vector store with document embeddings.
    
    Attempts to load existing vector store if available, unless force_rebuild is True
    If no existing store found or force_rebuild enabled, builds new store from documents.
    """
    try:
        embedding = HuggingFaceEmbeddings()
        should_load = os.path.exists(persist_directory) and not force_rebuild
        if should_load:
            try:
                vector_store = Chroma(
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    embedding_function=embedding,
                )
                logger.info("An existing vector store was loaded.")
                return vector_store
            except Exception as e:
                logger.error(f"Error loading existing vector store: {e}. Building new one.")
                should_load = False
        if not should_load:
            if documents is None:
                raise ValueError("Documents are required when building a new vector store.")
            vector_store = Chroma.from_documents(
                documents=documents,
                collection_name=collection_name,
                embedding=embedding,
                persist_directory=persist_directory,
            )
            logger.info("A new vector store was built.")
            return vector_store
    except Exception as e:
        raise ValueError(f"Error building/loading vector store: {e}.")
  

def load_documents(
    path: Path = Path("data"),
) -> List[Document]:
    """
    Load PDF documents from specified directory using PyPDFDirectoryLoader.
    """
    try:
        loader = PyPDFDirectoryLoader(
            path=path,
        )
        documents = loader.load()
        logger.info("Documents were loaded.")
        return documents
    except Exception as e:
        raise ValueError(f"Error loading documents: {e}.")
  

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into chunks using recursive character text splitting.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_documents = text_splitter.split_documents(
            documents=documents,
        )
        logger.info("Documents were split.")
        return split_documents
    except Exception as e:
        raise ValueError(f"Error splitting documents: {e}.")