import streamlit as st
import os
from utils import load_uploaded_file, apply_changes, undo_changes, clean_dataframe_for_display
from agent_manager import AgentManager
from analysis_agent import create_analysis_agent

def initialize_session_state():
    """Initialize session state for the application."""
    from utils import init_session_state
    init_session_state()
    
    # Initialize session state for analysis console
    if "console_log" not in st.session_state:
        st.session_state.console_log = []

def setup_agents(model_choice):
    """Initialize and setup AI agents."""
    # Initialize agent manager
    if "agent_manager" not in st.session_state or st.session_state.get("current_model") != model_choice:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            st.stop()
        st.session_state.agent_manager = AgentManager(api_key, model_choice)
        st.session_state.current_model = model_choice
    
    # Initialize analysis agent
    if "analysis_agent" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.session_state.analysis_agent = create_analysis_agent(
                api_key=api_key,
                model_name=model_choice,
                data_type="Cross-Sectional",
                df=st.session_state.df
            )

def render_model_selection():
    """Render model selection sidebar."""
    return st.sidebar.selectbox(
        "🤖 Choose AI Model:",
        ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "o1-preview"],
        index=0,
        help="Higher quality models = better results but more expensive"
    )

def render_file_upload_sidebar():
    """Render file upload section in sidebar."""
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

def render_control_buttons():
    """Render control buttons in sidebar."""
    with st.sidebar:
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

def render_chat_interface(declaration):
    """Render chat interface section."""
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
        
        # Process with appropriate agent
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # Use analysis agent if data type is declared, otherwise use general agent
                if declaration != "None" and "analysis_agent" in st.session_state:
                    try:
                        response = st.session_state.analysis_agent.run(prompt)
                        # Log the function call and result to the console
                        st.session_state.console_log.append({"call": f"analysis_agent.run('{prompt}')", "result": response})
                        # Check if response contains HTML image tag
                        if "<img" in response:
                            # Extract the text content before the image
                            text_parts = response.split("<img")
                            if text_parts[0].strip():
                                st.write(text_parts[0].strip())
                            
                            # Extract and display the image
                            img_start = response.find("<img")
                            img_end = response.find(">", img_start) + 1
                            img_html = response[img_start:img_end]
                            
                            # Display the image using st.markdown with unsafe_allow_html
                            st.markdown(img_html, unsafe_allow_html=True)
                            
                            # Display any text after the image
                            if len(text_parts) > 1:
                                after_img = text_parts[1].split(">", 1)
                                if len(after_img) > 1 and after_img[1].strip():
                                    st.write(after_img[1].strip())
                        else:
                            st.write(response)
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        # Fallback to general agent
                        response = st.session_state.agent_manager.process_query(prompt, st.session_state.df)
                        st.write(response)
                else:
                    # Use general agent for data preprocessing
                    response = st.session_state.agent_manager.process_query(prompt, st.session_state.df)
                    st.write(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

def render_data_preview():
    """Render data preview section."""
    st.subheader("📊 Current Data Preview")
    
    # Display basic info
    rows, cols = st.session_state.df.shape
    st.write(f"**Shape**: {rows} rows × {cols} columns")
    
    # Display data
    try:
        clean_df = clean_dataframe_for_display(st.session_state.df.head(20))
        st.dataframe(clean_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Display issue with dataframe: {str(e)}")
        # Fallback: show as text
        st.text(str(st.session_state.df.head(20)))

def render_welcome_screen():
    """Render welcome screen when no data is loaded."""
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