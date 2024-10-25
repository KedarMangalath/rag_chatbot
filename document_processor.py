import os
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    _instance = None
    _is_initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DocumentProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: Optional[str] = None,
        offline_mode: bool = False
    ):
        """
        Initialize DocumentProcessor with specified embedding model.
        Uses singleton pattern to maintain one instance.
        """
        if self._is_initialized:
            return
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Set up cache folder
        if cache_folder:
            cache_path = Path(cache_folder)
        else:
            cache_path = Path.home() / '.cache' / 'huggingface'
        
        # Create cache directory with proper permissions
        os.makedirs(str(cache_path), mode=0o777, exist_ok=True)
        
        # Set environment variable to disable symlinks warning
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        try:
            self.embeddings = self._initialize_embeddings(
                model_name,
                str(cache_path),
                offline_mode
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
            
        self.vectordb = None
        self._is_initialized = True

    def _initialize_embeddings(self, model_name: str, cache_folder: str, offline_mode: bool) -> HuggingFaceEmbeddings:
        """Initialize embeddings with retry logic and offline mode support."""
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder,
                model_kwargs={
                    'device': 'cpu',
                    'local_files_only': offline_mode
                },
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            if offline_mode:
                logger.error(f"Failed to load model in offline mode. Ensure model is cached in: {cache_folder}")
            raise

    def load_single_document(self, file_path: str) -> List:
        """Load a single document based on file type."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise

    def process_file(
        self,
        uploaded_file,
        persist_directory: str = "./chroma_db",
        force_reload: bool = False
    ) -> Optional[Chroma]:
        """
        Process an uploaded file and return a vector store.
        Uses cached vectordb if available unless force_reload is True.
        """
        # Return existing vectordb if available and reload not forced
        if self.vectordb is not None and not force_reload:
            return self.vectordb
            
        temp_file_path = None
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            # Load and split the document
            documents = self.load_single_document(temp_file_path)
            texts = self.text_splitter.split_documents(documents)
            
            # Ensure persist directory exists with proper permissions
            os.makedirs(persist_directory, mode=0o777, exist_ok=True)
            
            # Create and persist vector store
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            self.vectordb.persist()
            logger.info(f"Successfully created vector store in {persist_directory}")
            return self.vectordb
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return None
            
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    def get_vectordb(self) -> Optional[Chroma]:
        """Get the current vector store instance."""
        return self.vectordb