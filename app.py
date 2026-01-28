import streamlit as st
from dotenv import load_dotenv

from src.pdf_loader import load_pdf_text
from src.vector_store import create_vector_store
from src.rag_chain import create_rag_chain

load_dotenv()

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with PDF using OpenAI")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"]
)

if uploaded_file:
    try:
        with st.spinner("Reading PDF..."):
            pdf_text = load_pdf_text(uploaded_file)

        with st.spinner("Creating embeddings..."):
            vectorstore = create_vector_store(pdf_text)
            retriever = vectorstore.as_retriever()

        rag_chain = create_rag_chain(retriever)

        st.success("PDF processed successfully!")

        question = st.text_input("Ask a question from the PDF")

        if question:
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(question)
                st.markdown("### 💡 Answer")
                st.write(answer.content)

    except Exception as e:
        st.error(str(e))
