import streamlit as st
from document_parser import load_course_materials
from agent import ask_study_agent # Assuming you have this implemented

# --- Page Configuration ---
st.set_page_config(page_title="Calculemus Study Agent", page_icon="🎓", layout="wide")

st.title("🎓 Calculemus: Personalized Study Agent")
st.markdown("Welcome to the Friday Mini-Hack! Upload your course materials in the sidebar, then start hacking.")

# --- State Management ---
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_context" not in st.session_state:
    st.session_state.selected_context = ""

# --- Sidebar UI: Information Retrieval ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    
    # Provide a file uploader that accepts multiple files
    uploaded_files = st.file_uploader(
        "Upload course files or drag a folder here", 
        type=['pdf', 'md', 'txt', 'csv'], 
        accept_multiple_files=True
    )
    
    # Process button
    if st.button("Process Uploaded Materials") and uploaded_files:
        with st.spinner(f"Parsing {len(uploaded_files)} files..."):
            # Pass the list of uploaded file objects to our updated parser
            docs = load_course_materials(uploaded_files)
            st.session_state.documents = docs
        st.success(f"Successfully loaded {len(docs)} documents!")
    
    # Let the user pick which file to chat with
    if st.session_state.documents:
        selected_file = st.selectbox(
            "Select a file to set as context:", 
            options=["All Materials (Caution: Large!)"] + list(st.session_state.documents.keys())
        )
        
        # [EXTENSION POINT]
        # Currently, it just dumps the whole file text as context (limited to 8000 chars)... How can we optimize this?
        if selected_file == "All Materials (Caution: Large!)":
            st.session_state.selected_context = "\n".join(st.session_state.documents.values())[:8000] 
        else:
            st.session_state.selected_context = st.session_state.documents[selected_file][:8000]
            
        st.caption(f"Context size: {len(st.session_state.selected_context)} characters")

# --- Main UI: Chat Interface ---
# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask a question, generate flashcards, or extract deadlines..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            # Make sure ask_study_agent is defined elsewhere!
            response = ask_study_agent(prompt, context=st.session_state.selected_context)
            st.markdown(response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})