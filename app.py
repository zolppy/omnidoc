import streamlit as st
from utils.llm import build_model
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.rag import load_documents, split_documents, build_or_load_vector_store

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = None

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("📚 RAG Chatbot with Groq + LangChain")

def init_chain():
    documents = load_documents()
    split = split_documents(documents)
    vector_store = build_or_load_vector_store(documents=split)

    retriever = vector_store.as_retriever()
    llm = build_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Use *only* the following context from the document to answer accurately:\n\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
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
    return conversational_chain

with st.sidebar:
    st.header("⚙️ Options")
    if st.button("Initialize / Reload Model"):
        st.session_state.conversational_chain = init_chain()
        st.success("Model and retriever initialized!")

    if st.button("🗑️ Restart Chat"):
        st.session_state.history = InMemoryChatMessageHistory()
        st.success("Chat history cleared!")

if st.session_state.conversational_chain:
    user_input = st.chat_input("Ask me something about the documents...")
    if user_input:
        response = st.session_state.conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}},
        )

    for msg in st.session_state.history.messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.write(msg.content)
        else:
            with st.chat_message("assistant"):
                st.write(msg.content)
else:
    st.info("Click **Initialize / Reload Model** in the sidebar to get started.")
