# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.schema import Document
# from secret_key import grok_api_key
# import os
# import tempfile
# from llm_config import get_llm

# os.environ["GROQ_API_KEY"] = grok_api_key

# st.title("📄 Document-Based LLM ChatBot")
# st.write("Upload PDFs to ask questions about their content!")


# # Initialize the model
# @st.cache_resource
# def get_qa_llm():
#     return ChatGroq(
#         groq_api_key=grok_api_key,
#         model_name="llama3-70b-8192",
#         temperature=0.1
#     )


# # Initialize embeddings
# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )


# # Improved QA prompt
# qa_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#     You are a helpful assistant that answers questions based on the provided context. 

#     Use the following context to answer the question as accurately as possible. 
#     If the exact answer is not in the context but you can infer it from related information, make that clear.
#     If you truly cannot find any relevant information, then say "I cannot find this information in the provided document."

#     Context: {context}

#     Question: {question}

#     Answer: """
# )


# # Helper functions
# def process_pdf(uploaded_file):
#     """Process uploaded PDF file"""
#     try:
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name

#         # Load PDF
#         loader = PyPDFLoader(tmp_file_path)
#         documents = loader.load()

#         # Clean up temp file
#         os.unlink(tmp_file_path)

#         return documents
#     except Exception as e:
#         st.error(f"Error processing PDF: {str(e)}")
#         return None


# def create_vector_store(documents):
#     """Create vector store from documents with improved chunking"""
#     try:
#         # Split documents with better parameters
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500,  # Increased chunk size
#             chunk_overlap=300,  # Increased overlap
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
#         )
#         chunks = text_splitter.split_documents(documents)

#         # Show chunk information
#         st.info(f"Created {len(chunks)} text chunks for processing")

#         # Create embeddings
#         embeddings = get_embeddings()

#         # Create vector store
#         vector_store = FAISS.from_documents(chunks, embeddings)

#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {str(e)}")
#         return None


# # UI Layout
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# documents = None
# vector_store = None

# if uploaded_file is not None:
#     st.success(f"Uploaded: {uploaded_file.name}")

#     if st.button("Process PDF"):
#         with st.spinner("Processing PDF..."):
#             documents = process_pdf(uploaded_file)

#             if documents:
#                 st.success(f"✅ PDF processed! Found {len(documents)} pages.")

#                 # Create vector store
#                 with st.spinner("Creating searchable index..."):
#                     vector_store = create_vector_store(documents)
#                     if vector_store:
#                         st.session_state.vector_store = vector_store
#                         st.session_state.documents_loaded = True
#                         st.success("✅ Document indexed and ready for questions!")


# # Q&A Interface
# st.markdown("---")
# st.header("Ask Questions")

# if 'documents_loaded' in st.session_state and st.session_state.documents_loaded:

#     # Initialize chat history for Q&A
#     if "qa_messages" not in st.session_state:
#         st.session_state.qa_messages = []

#     # Display chat history
#     for message in st.session_state.qa_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Question input
#     if question := st.chat_input("Ask a question about the document..."):
#         # Add user question
#         st.session_state.qa_messages.append({"role": "user", "content": question})
#         with st.chat_message("user"):
#             st.markdown(question)

#         # Generate answer
#         with st.chat_message("assistant"):
#             with st.spinner("Searching for answer..."):
#                 try:
#                     # Create QA chain with better retrieval
#                     llm = get_qa_llm()
#                     qa_chain = RetrievalQA.from_chain_type(
#                         llm=llm,
#                         chain_type="stuff",
#                         retriever=st.session_state.vector_store.as_retriever(
#                             search_type="similarity",
#                             search_kwargs={"k": 5}  # Retrieve more chunks
#                         ),
#                         chain_type_kwargs={"prompt": qa_prompt},
#                         return_source_documents=True
#                     )

#                     # Get answer
#                     response = qa_chain.invoke({"query": question})
#                     answer = response["result"]

#                     st.markdown(answer)

#                     # Show source information
#                     with st.expander("📄 Source Information"):
#                         for i, doc in enumerate(response.get("source_documents", [])):
#                             st.text(f"Chunk {i + 1}: {doc.page_content[:200]}...")

#                     st.session_state.qa_messages.append({"role": "assistant", "content": answer})

#                 except Exception as e:
#                     error_msg = f"Error generating answer: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.qa_messages.append({"role": "assistant", "content": error_msg})

#     # Clear chat button
#     if st.button("Clear Q&A History"):
#         st.session_state.qa_messages = []
#         st.rerun()

# else:
#     st.info("👆 Please upload a PDF first to start asking questions!")

# # Testing section
# if 'documents_loaded' in st.session_state and st.session_state.documents_loaded:
#     with st.expander("🔍 Debug Information"):
#         if st.button("Chunk Differenciation"):
#             test_query = "vision mission"
#             retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
#             docs = retriever.get_relevant_documents(test_query)
#             st.write(f"Found {len(docs)} relevant chunks for '{test_query}':")
#             for i, doc in enumerate(docs):
#                 st.text(f"Chunk {i + 1}: {doc.page_content[:300]}...")

# # Example questions
# with st.expander("💡 Example Questions"):
#     st.markdown("""
#     **For Educational Institutions:**
#     - "What is the vision of the institution?"
#     - "What courses are offered?"
#     - "What is the admission process?"
#     - "What are the objectives of the program?"

#     **For Research Papers:**
#     - "What is the main conclusion of this study?"
#     - "What methodology was used?"
#     - "What are the key findings?"

#     **For Articles:**
#     - "Summarize the main points"
#     - "Who are the key people mentioned?"
#     - "What are the implications discussed?"
#     """)

import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from secret_key import grok_api_key
import os
import tempfile

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📄",
    layout="wide"
)

def chatbot_page():
    """
    This function contains the entire chatbot application.
    """
    st.sidebar.title("Logout")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.title("📄 Document-Based LLM ChatBot")
    st.write("Upload a PDF document and ask questions about its content. The chatbot will use an LLM to find the answers for you.")

    # --- Caching for Resource-Intensive Functions ---
    @st.cache_resource
    def get_qa_llm():
        """
        Initializes and caches the Language Model for Question Answering.
        """
        try:
            return ChatGroq(
                groq_api_key=grok_api_key,
                model_name="llama3-70b-8192",
                temperature=0.1
            )
        except Exception as e:
            st.error(f"Failed to initialize the language model: {e}")
            return None

    @st.cache_resource
    def get_embeddings():
        """
        Initializes and caches the text embedding model.
        """
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Failed to initialize embeddings model: {e}")
            return None

    # --- Prompt Template ---
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant designed to answer questions based on a provided document.

        Use the following context to answer the question as accurately and completely as possible.
        If the answer is not directly stated in the context, but can be reasonably inferred, please state that your answer is an inference.
        If you cannot find any relevant information in the context to answer the question, clearly say: "I cannot find this information in the provided document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # --- Core Functions ---
    def process_pdf(uploaded_file):
        """
        Processes an uploaded PDF file, extracts its text, and returns it as a list of documents.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            os.unlink(tmp_file_path)
            return documents
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            return None

    def create_vector_store(documents):
        """
        Takes a list of documents, splits them into chunks, and creates a searchable vector store.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            st.info(f"The document was split into {len(chunks)} text chunks for analysis.")
            embeddings = get_embeddings()
            if embeddings is None:
                return None
            vector_store = FAISS.from_documents(chunks, embeddings)
            return vector_store
        except Exception as e:
            st.error(f"Failed to create the vector store: {e}")
            return None

    # --- Main UI Layout ---
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success(f"Successfully uploaded: {uploaded_file.name}")
        if st.button("Process Document"):
            with st.spinner("Processing document, please wait..."):
                documents = process_pdf(uploaded_file)
                if documents:
                    st.success(f"✅ Document processed! Found {len(documents)} pages.")
                    with st.spinner("Creating searchable index..."):
                        vector_store = create_vector_store(documents)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.documents_loaded = True
                            st.success("✅ Document is indexed and ready for questions!")
                        else:
                            st.error("Could not create the document index.")

    st.markdown("---")

    # --- Q&A Interface ---
    st.header("Ask Questions About Your Document")

    if 'documents_loaded' in st.session_state and st.session_state.documents_loaded:
        if "qa_messages" not in st.session_state:
            st.session_state.qa_messages = []

        for message in st.session_state.qa_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("Ask a question..."):
            st.session_state.qa_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching for the answer in the document..."):
                    try:
                        llm = get_qa_llm()
                        if llm is None:
                            st.error("Language model is not available.")
                        else:
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=st.session_state.vector_store.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": 5}
                                ),
                                chain_type_kwargs={"prompt": qa_prompt},
                                return_source_documents=True
                            )
                            response = qa_chain.invoke({"query": question})
                            answer = response.get("result", "No answer found.")
                            st.markdown(answer)
                            with st.expander("📄 View Sources"):
                                source_docs = response.get("source_documents", [])
                                if source_docs:
                                    for i, doc in enumerate(source_docs):
                                        st.write(f"**Source Chunk {i+1}:**")
                                        st.info(f"{doc.page_content[:300]}...")
                                else:
                                    st.write("No source chunks found for this answer.")
                            st.session_state.qa_messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"An error occurred while generating the answer: {e}"
                        st.error(error_msg)
                        st.session_state.qa_messages.append({"role": "assistant", "content": error_msg})

        if st.session_state.qa_messages:
            if st.button("Clear Chat History"):
                st.session_state.qa_messages = []
                st.rerun()
    else:
        st.info("👆 Please upload and process a PDF document to begin.")

    with st.expander("💡 Example Questions You Could Ask"):
        st.markdown("""
        - "What is the main purpose of this document?"
        - "Summarize the key findings from the study."
        - "What are the names of the people mentioned in the introduction?"
        - "Explain the methodology used for the experiment."
        """)

def login_page():
    """
    Displays the login page for the user.
    """
    st.title("Login")
    st.write("Please enter your credentials to access the chatbot.")

    # Static credentials for demonstration
    credentials = {"user": "password"}

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in credentials and credentials[username] == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")

def main():
    """
    Main function to run the Streamlit app.
    It handles the routing between the login page and the chatbot page.
    """
    # Set the API key for the Groq service
    os.environ["GROQ_API_KEY"] = grok_api_key

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        chatbot_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
