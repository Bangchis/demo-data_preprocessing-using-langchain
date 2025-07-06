import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from utils import (
    init_session_state, 
    load_uploaded_file, 
    apply_changes, 
    undo_changes,
    clean_dataframe_for_display  # ← THÊM DÒNG NÀY
)
from agent_manager import AgentManager

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Data Preprocessing Chat",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Data Preprocessing Chat MVP")
    st.markdown("Upload your CSV/XLSX files and chat with AI to preprocess your data!")
    
    # Initialize session state
    init_session_state()
    
    # Model selection FIRST (before using it)
    model_choice = st.sidebar.selectbox(
        "🤖 Choose AI Model:",
        ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "o1-preview"],
        index=0,
        help="Higher quality models = better results but more expensive"
    )
    
    # Initialize agent manager
    if "agent_manager" not in st.session_state or st.session_state.get("current_model") != model_choice:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            st.stop()
        st.session_state.agent_manager = AgentManager(api_key, model_choice)
        st.session_state.current_model = model_choice
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Choose CSV/XLSX files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.dfs:
                    df = load_uploaded_file(uploaded_file)
                    if df is not None:
                        st.session_state.dfs[uploaded_file.name] = df
                        st.success(f"✅ Loaded {uploaded_file.name}")
                        
                        # Set as main df if first file
                        if st.session_state.df is None:
                            st.session_state.df = df.copy()
        
        # Display loaded files
        if st.session_state.dfs:
            st.subheader("📊 Loaded Files")
            for filename in st.session_state.dfs.keys():
                rows, cols = st.session_state.dfs[filename].shape
                st.write(f"• **{filename}**: {rows} rows, {cols} cols")
        
        # Control buttons
        st.subheader("🎛️ Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply", disabled=st.session_state.df is None):
                apply_changes()
        
        with col2:
            if st.button("↩️ Undo", disabled=len(st.session_state.history) == 0):
                undo_changes()
        
        # Download button
        if st.session_state.df is not None:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    
    # Main content area
    if st.session_state.df is not None:
        # Chat interface
        st.subheader("💬 Chat with your data")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to do with your data?"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process with agent
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = st.session_state.agent_manager.process_query(
                        prompt, 
                        st.session_state.df
                    )
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Data preview
        st.subheader("📊 Current Data Preview")
        
        # Display basic info
        rows, cols = st.session_state.df.shape
        st.write(f"**Shape**: {rows} rows × {cols} columns")
        
        # Display data
        try:
            st.dataframe(st.session_state.df.head(20), use_container_width=True)
        except Exception as e:
            st.warning(f"Display issue with dataframe: {str(e)}")
            # Fallback: show as text
            st.text(str(st.session_state.df.head(20)))
        
        # Display data types
        with st.expander("📋 Data Types & Info"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                try:
                    st.write(st.session_state.df.dtypes)
                except:
                    st.text(str(st.session_state.df.dtypes))
            
            with col2:
                st.write("**Missing Values:**")
                try:
                    st.write(st.session_state.df.isnull().sum())
                except:
                    st.text(str(st.session_state.df.isnull().sum()))
    
    else:
        st.info("👆 Please upload at least one CSV or XLSX file to get started!")
        
        # Show example commands
        st.subheader("💡 Example Commands")
        examples = [
            "Show me basic statistics of the data",
            "Remove rows with missing values",
            "Fill missing values in column 'age' with the mean",
            "Merge all uploaded files on column 'id'",
            "Create a new column 'total' by adding column1 and column2",
            "Remove outliers from column 'price' using IQR method",
            "Group by 'category' and calculate mean of 'value'",
            "Convert 'date' column to datetime format"
        ]
        
        for example in examples:
            st.code(example)

if __name__ == "__main__":
    main()