import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Network Error: Could not connect to Google's embedding service. {e}")
        return False

def create_prompt(response_mode, is_rag=False):
    """Creates a highly specific and structured prompt based on the desired response style."""
    
    base_instruction = "When you provide code, you MUST wrap it in triple backticks (e.g., ```python ... ```) for proper formatting."

    # Specific style instructions
    if response_mode == "Concise":
        style_instruction = "You are an expert at summarization. Your response MUST be a single paragraph of 3-4 sentences only. Distill the answer to its absolute core essence."
    
    elif response_mode == "Detailed":
        style_instruction = (
            "Several paragraphs, at least 500 words. Response must follow a strict teaching-style structure with markdown headings. "
            "The tone should be like a university professor preparing teaching material.\n\n"
            "Structure:\n\n"
            "### 📌 Overview\n- Provide a clear and precise definition of the concept.\n- Explain the broader context (where it fits, why it matters).\n\n"
            "### [Extracted Heading 1]\n- Expand thoroughly on the section using clear explanations.\n- Use bullet points if needed.\n- Add step-by-step or algorithmic explanation if relevant.\n\n"
            "### [Extracted Heading 2]\n- Continue explaining in detail as per the document.\n\n"
            "### [Extracted Heading N]\n- Repeat for each key heading from the PDF or reference text.\n\n"
            "Finally, include a summary table:\n\n"
            "### Advantages vs Limitations\n- Present a balanced view in a simple two-column table, with one column for Advantages and another for Limitations.\n\n"
            "Purpose: For learners, researchers, or exam preparation. This ensures the response is a complete deep dive that mirrors the source document’s structure while teaching the material clearly.\n\n"
            "### Conclusion\n- Summarize the key points and their implications.\n\n"
        )
    
    else:  # Normal
        style_instruction = "You are a helpful AI assistant. Provide a clear and balanced answer in 3 paragraphs, approximately 3-5 sentences long in each paragraph. Start with a simple definition, then briefly explain its main purpose or function, and conclude with a common example."

    # Combine instructions based on context type (RAG vs. General)
    if is_rag:
        return (
            f"Your primary instruction is to answer based *only* on the provided context. "
            f"If the answer is not in the context, state that clearly and do not use outside knowledge. {base_instruction}\n\n"
            f"Your required output style is: {style_instruction}\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    else:
        return f"{base_instruction}\n\nYour required output style is: {style_instruction}"



def handle_user_query(prompt, response_mode, page, chat_history):
    if page == "Upload":
        if not os.path.exists("faiss_index"): return "Knowledge base not found. Please upload documents."
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(prompt, k=4)
            
            prompt_template_str = create_prompt(response_mode, is_rag=True)
            prompt_obj = PromptTemplate.from_template(prompt_template_str)
            
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
            chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt_obj)
            response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
            return response["output_text"]
        except Exception as e: return f"Error querying Gemini: {e}"
    elif page == "Chat":
        if not groq_client: return "Groq API client not initialized. Check your GROQ_API_KEY."
        
        system_prompt = create_prompt(response_mode, is_rag=False)
        messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": prompt}]
        
        try:
            response = groq_client.chat.completions.create(messages=messages, model="llama3-70b-8192")
            return response.choices[0].message.content
        except Exception as e: return f"Error from Groq API: {e}"
    return "Invalid page context."

# UI 
def render_ui():
    st.markdown("""
        <style>
            .stApp { background-color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E6EAF1; }
            .sidebar-header {
                display: flex; align-items: center; gap: 10px; font-size: 1.75rem;
                font-weight: 700; color: #1A73E8; padding: 1.5rem;
            }
            .nav-button-container {
                background-color: #F8F9FA; border-radius: 10px; margin: 0.5rem 0;
                border: 1px solid #F8F9FA; transition: all 0.2s;
            }
            .nav-button-container:hover { border-color: #1A73E8; background-color: #E7F1FF; }
            .nav-button-container.active { background-color: #1A73E8; border-color: #1A73E8; }
            .nav-button-container div[data-testid="stButton"] > button {
                background-color: transparent; color: #212529; border: none; text-align: left;
                padding: 0.75rem 1rem; font-size: 1.05rem; font-weight: 500;
                display: flex; align-items: center; gap: 10px;
            }
            .nav-button-container.active div[data-testid="stButton"] > button { color: white; }
            .chat-input-container {
                position: fixed; bottom: 0; left: 305px; right: 0;
                padding: 1rem 2rem; background-color: #FFFFFF;
                border-top: 1px solid #E0E0E0; z-index: 99;
            }
            .chat-toolbar { display: flex; justify-content: flex-end; margin-bottom: 0.5rem; }
            .main-content-wrapper { padding-bottom: 10rem; }
        </style>
    """, unsafe_allow_html=True)

def render_page(page_title, caption, history_key, page_context):
    st.title(page_title)
    st.caption(caption)
    
    with st.container():
        st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
        if page_context == "Upload":
            with st.container(border=True):
                st.subheader("Upload Your Knowledge Base")
                uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf", label_visibility="collapsed")
                if st.button("Process Documents", type="primary", use_container_width=True):
                    if uploaded_files:
                        with st.spinner("Processing..."):
                            if get_vector_store(get_text_chunks(get_pdf_text(uploaded_files))):
                                st.success("Documents processed successfully!")
                                st.session_state.upload_chat_history = []
                    else: st.warning("Please upload at least one PDF file.")
        
        if history_key not in st.session_state: st.session_state[history_key] = []
        show_chat = (page_context == "Chat") or (page_context == "Upload" and os.path.exists("faiss_index"))
        if show_chat:
            if page_context == "Upload": st.markdown("---"); st.subheader("Chat with Your Documents")
            for msg in st.session_state[history_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        st.markdown('<div class="chat-toolbar">', unsafe_allow_html=True)
        response_mode = st.selectbox(
            "Response Style", ("Normal", "Concise", "Detailed"),
            key=f"{history_key}_response_mode", label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if prompt := st.chat_input("Ask a question..."):
            st.session_state[history_key].append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                response = handle_user_query(prompt, response_mode, page_context, st.session_state[history_key])
                st.session_state[history_key].append({"role": "assistant", "content": response})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="NebuEye", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    render_ui()

    if "page" not in st.session_state: st.session_state.page = "Chat"

    with st.sidebar:
        st.markdown('<div class="sidebar-header">🤖 NebuEye</div>', unsafe_allow_html=True)
        if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.upload_chat_history = []
            st.session_state.page = "Chat"
            st.rerun()
        st.markdown("---")
        st.subheader("Navigation")
        
        is_chat_active = st.session_state.page == "Chat"
        with st.container():
            st.markdown(f'<div class="nav-button-container {"active" if is_chat_active else ""}">', unsafe_allow_html=True)
            if st.button("💬 General Chat", use_container_width=True, key="nav_chat"):
                st.session_state.page = "Chat"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        is_upload_active = st.session_state.page == "Upload"
        with st.container():
            st.markdown(f'<div class="nav-button-container {"active" if is_upload_active else ""}">', unsafe_allow_html=True)
            if st.button("📄 Document Chat", use_container_width=True, key="nav_upload"):
                st.session_state.page = "Upload"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.page == "Upload":
        render_page("Chat with Documents", "Powered by Google Gemini", "upload_chat_history", "Upload")
    else:
        render_page("General Chat", "Powered by Groq Llama 3", "chat_history", "Chat")

if __name__ == "__main__":
    main()