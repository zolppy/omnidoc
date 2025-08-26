# OmniDoc - RAG Chatbot with Groq + LangChain

A Retrieval-Augmented Generation (RAG) chatbot application that leverages Groq's high-performance language models and LangChain framework to provide intelligent responses based on document content.

## 📋 Overview

This project implements a document-based question answering system that:
- Loads and processes PDF documents from a specified directory
- Creates vector embeddings for semantic search
- Utilizes Groq's LLM for generating responses
- Maintains conversation history for contextual interactions
- Provides a clean Streamlit web interface

## 🚀 Features

- **Document Processing**: Automatic loading and chunking of PDF documents
- **Vector Store**: ChromaDB integration for efficient document retrieval
- **LLM Integration**: Groq API connection with configurable models
- **Conversation Memory**: Maintains chat history throughout sessions
- **User-Friendly Interface**: Streamlit-based web UI with sidebar controls
- **Logging System**: Comprehensive logging with file rotation and error tracking

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd rag-chatbot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

## 📁 Project Structure

```
project/
├── app.py                 # Main Streamlit application
├── logger.py             # Logging configuration and setup
├── rag.py               # Document processing and vector store management
├── llm.py               # LLM model initialization
├── data/                # Directory for PDF documents (create this)
├── logs/                # Auto-generated log files directory
└── vector_store/        # Auto-generated vector store directory
```

## ⚙️ Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key (required)

### Optional Parameters
- **Model Selection**: Change the default model in `llm.py` (default: "llama3-70b-8192")
- **Document Directory**: Modify the path in `rag.py` (default: "data/")
- **Chunk Settings**: Adjust chunk size and overlap in `split_documents()`
- **Logging**: Configure log levels and file paths in `logger.py`

## 🎯 Usage

1. Place your PDF documents in the `data/` directory
2. Run the application:
```bash
streamlit run app.py
```
3. Open your browser to the provided local URL (typically http://localhost:8501)
4. Click "Initialize / Reload Model" in the sidebar to process documents
5. Start chatting with your documents!

### Sidebar Options
- **Initialize/Reload Model**: Processes documents and sets up the vector store
- **Restart Chat**: Clears conversation history while keeping the model loaded

## 🔧 Components

### Logger Module
- Configurable logging to both console and files
- Rotating file handlers for main logs and error logs
- Custom formatting with timestamps, log levels, and source information

### RAG Module
- Document loading from PDF directories
- Text splitting with configurable chunk parameters
- Vector store management with persistent storage
- Support for both building new and loading existing vector stores

### LLM Module
- Groq API integration
- Configurable model selection and temperature
- Error handling for model initialization

### Application
- Streamlit-based web interface
- Conversation history management
- Retrieval chain implementation with context-aware responses

## 📊 Logging

The application generates logs in the `logs/` directory:
- `app.log`: General application logs with DEBUG level
- `errors.log`: Error-specific logs with rotating daily retention

## 🚨 Troubleshooting

### Common Issues
1. **GROQ_API_KEY not set**: Ensure you've set the environment variable
2. **No documents found**: Place PDF files in the `data/` directory
3. **Vector store errors**: Use "force_rebuild=True" or delete the vector_store directory to rebuild

### Debug Mode
Enable debug logging by modifying the console_level parameter in `logger.py`:
```python
console_level=logging.DEBUG
```

## 📝 License

This project is available for use and modification.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For questions or issues, please check the logs in the `logs/` directory or open an issue in the project repository.
