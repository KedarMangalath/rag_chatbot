import os
import logging
from typing import Tuple, List
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self, vector_store):
        """Initialize ChatHandler with a vector store."""
        # Check for GROQ API key
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        try:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=ChatGroq(
                    temperature=0.1,
                    model_name="mixtral-8x7b-32768",
                    api_key=self.api_key  # Explicitly pass API key
                ),
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Error initializing ConversationalRetrievalChain: {str(e)}")
            raise
    
    def generate_response(self, user_input: str) -> Tuple[str, List[Document]]:
        """
        Generate a response to user input.
        
        Args:
            user_input (str): The user's question or input
            
        Returns:
            Tuple[str, List[Document]]: The response and source documents
        """
        try:
            response = self.chain.invoke({"question": user_input})
            return response["answer"], response["source_documents"]
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error processing your request.", []