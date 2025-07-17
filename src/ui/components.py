import streamlit as st
import os
from src.core.utils import load_uploaded_file, load_uploaded_file_preserve_types, apply_changes, undo_changes, clean_dataframe_for_display, get_current_dataframe_for_display
# from src.ui.data_view_controller import get_current_dataframe_for_agent  # Simplified: removed complex view controller
from src.agents.react_agent import AgentManager
from src.agents.analysis_agent import create_analysis_agent
from src.core.backup_manager import backup_manager

def initialize_session_state():
    """Initialize session state for the application."""
    from src.core.utils import init_session_state
    init_session_state()
    
    # Initialize session state for analysis console
    if "console_log" not in st.session_state:
        st.session_state.console_log = []
    
    # Initialize session state for ReAct agent
    if "react_chat_history" not in st.session_state:
        st.session_state.react_chat_history = []
    
    # Initialize session state for execution logging
    if "execution_log" not in st.session_state:
        st.session_state.execution_log = []
    
    # Initialize session state for checkpoints
    if "checkpoints" not in st.session_state:
        st.session_state.checkpoints = []
    
    if "checkpoint_index" not in st.session_state:
        st.session_state.checkpoint_index = -1
    
    # Initialize session state for web search
    if "web_search_log" not in st.session_state:
        st.session_state.web_search_log = []
    
    # Initialize current time for logging
    if "current_time" not in st.session_state:
        import datetime
        st.session_state.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize backup system session state
    backup_manager._ensure_session_state()

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
                    # Use new type-preserving loader with raw data support
                    df_raw, df_original, df_display = load_uploaded_file_preserve_types(uploaded_file)
                    if df_raw is not None and df_original is not None and df_display is not None:
                        # Store all three versions
                        st.session_state.dfs_raw[uploaded_file.name] = df_raw
                        st.session_state.dfs_original[uploaded_file.name] = df_original
                        st.session_state.dfs_display[uploaded_file.name] = df_display
                        st.session_state.dfs[uploaded_file.name] = df_display  # Legacy compatibility
                        
                        st.success(f"✅ Loaded {uploaded_file.name} (raw + processed + display versions)")
                        
                        # Set as main df if first file
                        if st.session_state.df is None:
                            st.session_state.df_raw = df_raw.copy()
                            st.session_state.df_original = df_original.copy()
                            st.session_state.df_display = df_display.copy()
                            st.session_state.df = df_display.copy()  # Legacy compatibility
        
        # Display loaded files
        if st.session_state.dfs:
            st.subheader("📊 Loaded Files")
            for filename in st.session_state.dfs.keys():
                rows, cols = st.session_state.dfs[filename].shape
                
                # Show comprehensive data information
                if filename in st.session_state.get("dfs_original", {}):
                    df_orig = st.session_state.dfs_original[filename]
                    numeric_cols = len(df_orig.select_dtypes(include=['number']).columns)
                    datetime_cols = len(df_orig.select_dtypes(include=['datetime']).columns)
                    bool_cols = len(df_orig.select_dtypes(include=['bool']).columns)
                    
                    st.write(f"• **{filename}**: {rows} rows, {cols} cols")
                    st.write(f"  📊 Processed: {numeric_cols} numeric, {datetime_cols} datetime, {bool_cols} boolean")
                    
                    # Show raw data info if available
                    if filename in st.session_state.get("dfs_raw", {}):
                        df_raw = st.session_state.dfs_raw[filename]
                        raw_object_cols = len(df_raw.select_dtypes(include=['object']).columns)
                        st.write(f"  📄 Raw: {raw_object_cols} object, {cols - raw_object_cols} non-object")
                else:
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
    
    # Get the current DataFrame for display (use df_original if available for better info)
    current_df_for_display = get_current_dataframe_for_display()
    original_df = st.session_state.get("df_original", None)
    
    if current_df_for_display is not None:
        # Display basic info
        rows, cols = current_df_for_display.shape
        st.write(f"**Shape**: {rows} rows × {cols} columns")
        
        # Show data types summary if we have original data
        if original_df is not None:
            dtype_counts = original_df.dtypes.value_counts()
            dtype_summary = ", ".join([f"{count} {dtype}" for dtype, count in dtype_counts.items()])
            st.write(f"**Data Types**: {dtype_summary}")
        
        # Display data
        try:
            # Show first 20 rows
            st.dataframe(current_df_for_display.head(20), use_container_width=True)
            
            # Show processing information if available
            if st.session_state.get("processing_pipeline"):
                st.info("💡 **Tip**: CodeRunner operates on the processed data with proper data types.")
                
        except Exception as e:
            st.warning(f"Display issue with dataframe: {str(e)}")
            # Fallback: show as text
            try:
                st.text(str(current_df_for_display.head(20)))
            except:
                st.error("Unable to display dataframe content.")
    else:
        st.info("No data available to preview.")

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