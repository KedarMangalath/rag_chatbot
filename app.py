import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.chat_handler import ChatHandler

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_handler" not in st.session_state:
        st.session_state.chat_handler = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Document Chat Assistant")
    
    # Check for GROQ API key
    if not os.getenv('GROQ_API_KEY'):
        st.error("""
        GROQ API key not found! Please set up your API key:
        1. Create a .env file in your project directory
        2. Add the line: GROQ_API_KEY=your_api_key_here
        3. Restart the application
        """)
        st.stop()
    
    initialize_session_state()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                # Initialize document processor
                doc_processor = DocumentProcessor(
                    cache_folder="./model_cache",
                    offline_mode=False
                )
                
                # Process the file and create vector store
                vector_store = doc_processor.process_file(
                    uploaded_file,
                    persist_directory="./chroma_db"
                )
                
                if vector_store:
                    try:
                        st.session_state.chat_handler = ChatHandler(vector_store)
                        st.success("Document processed successfully!")
                    except ValueError as e:
                        st.error(f"Error initializing chat handler: {str(e)}")
                        st.stop()
                else:
                    st.error("Error processing document.")
                    return
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                return
    
    # Chat interface
    if st.session_state.chat_handler:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if user_input := st.chat_input("Ask a question about your document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.chat_handler.generate_response(user_input)
                    st.markdown(response)
                    
                    # Display sources if available
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.markdown(f"From page {source.metadata.get('page', 'N/A')}:")
                                st.markdown(source.page_content)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()