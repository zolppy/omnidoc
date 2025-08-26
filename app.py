import streamlit as st
from utils.llm import build_model
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.rag import load_documents, split_documents, build_or_load_vector_store
import time

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("📚 RAG Chatbot with Groq + LangChain")

def init_chain():
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading documents...")
        documents = load_documents()
        progress_bar.progress(25)
        
        status_text.text("Splitting documents...")
        split = split_documents(documents)
        progress_bar.progress(50)
        
        status_text.text("Building vector store...")
        vector_store = build_or_load_vector_store(documents=split)
        progress_bar.progress(75)
        
        status_text.text("Initializing model...")
        retriever = vector_store.as_retriever()
        llm = build_model()

        # Enhanced prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """# Role & Context
                You are an expert research assistant tasked with answering questions based *exclusively* on the provided document context.

                ## Instructions
                1. **Context-Based Responses**: Use ONLY the information from the provided context to answer questions
                2. **Accuracy & Precision**: Provide accurate, well-structured answers with specific details when available
                3. **Uncertainty Handling**: If the context doesn't contain sufficient information, clearly state what you cannot answer and why
                4. **Citation Readiness**: Structure your response to make it easy to trace information back to the source material
                5. **Conversation Flow**: Maintain natural conversation while adhering strictly to the document content

                ## Response Guidelines
                - Begin with a clear, direct answer to the question
                - Provide supporting evidence from the context when relevant
                - Use bullet points or numbered lists for multiple items or steps
                - Maintain a professional yet approachable tone
                - Avoid speculation, assumptions, or external knowledge
                - If context is insufficient, explain what information is missing

                ## Context Information:
                {context}

                ## Current Conversation:"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        conversational_chain = RunnableWithMessageHistory(
            retrieval_chain,
            lambda _: st.session_state.history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return conversational_chain
        
    except Exception as e:
        st.sidebar.error(f"Initialization failed: {str(e)}")
        raise

with st.sidebar:
    st.header("⚙️ Options")
    
    if st.button("Initialize / Reload Model", type="primary"):
        try:
            st.session_state.conversational_chain = init_chain()
            st.session_state.initialized = True
            st.success("✅ Model and retriever initialized successfully!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")

    if st.button("🗑️ Restart Chat"):
        st.session_state.history = InMemoryChatMessageHistory()
        st.success("💬 Chat history cleared!")

    st.sidebar.divider()
    st.sidebar.subheader("Status")
    if st.session_state.initialized:
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.warning("⚠️ Not Initialized")

if st.session_state.conversational_chain:
    for msg in st.session_state.history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)
    
    user_input = st.chat_input("Ask me something about the documents...")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversational_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": "default"}},
                    )
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("👈 Click **Initialize / Reload Model** in the sidebar to get started.")
    st.info("💡 Make sure to:")
    st.info("1. Place PDF documents in the `data/` directory")
    st.info("2. Set your GROQ_API_KEY environment variable")
    st.info("3. Have an internet connection for model access")