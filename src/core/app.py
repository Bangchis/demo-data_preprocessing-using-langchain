import streamlit as st
import pandas as pd
import os
from src.core.utils import clean_dataframe_for_display
from src.agents.react_agent import AgentManager
from src.utils.token_manager import get_token_manager

from src.analysis.ui import get_analysis_toolbar
from src.analysis.forms import prompt_user_for_inputs
from src.analysis.declaration import (
    render_data_declaration_section,
    handle_panel_data_declaration,
    handle_time_series_declaration,
    handle_cross_sectional_declaration
)
from src.analysis.quality import render_data_types_section, render_data_quality_checks
from src.ui.components import (
    initialize_session_state,
    setup_agents,
    render_model_selection,
    render_file_upload_sidebar,
    render_control_buttons,
    render_data_preview,
    render_welcome_screen
)
from src.ui.backup_components import (
    render_backup_control_panel,
    render_backup_status_indicator,
    render_backup_notification
)
from src.ui.data_quality_dashboard import render_data_quality_dashboard
from src.ui.data_view_controller import (
    render_data_view_selector,
    render_processing_pipeline,
    render_agent_data_source_warning,
    render_data_view_comparison
)


def setup_react_agent(model_choice):
    """Setup ReAct agent with tools"""
    if "react_agent_manager" not in st.session_state or st.session_state.get("current_react_model") != model_choice:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            st.stop()
        st.session_state.react_agent_manager = AgentManager(api_key, model_choice)
        st.session_state.current_react_model = model_choice
        st.success(f"✅ ReAct Agent initialized with {model_choice}")


def render_react_chat_interface():
    """Render ReAct agent chat interface"""
    st.subheader("💬 ReAct Agent Chat")
    st.markdown("**Tính năng mới:** Agent thông minh với khả năng tìm kiếm web và tools chuyên dụng!")
    
    # Show available tools
    if 'react_agent_manager' in st.session_state:
        with st.expander("🔧 Available Tools"):
            tools = st.session_state.react_agent_manager.get_tool_descriptions()
            
            # Group tools by category
            core_tools = {}
            exploration_tools = {}
            web_tools = {}
            
            for tool_name, description in tools.items():
                if tool_name in ['CodeRunner', 'Undo', 'Redo', 'ExecutionLog']:
                    core_tools[tool_name] = description
                elif tool_name in ['WebSearch', 'PandasHelp', 'DataScienceHelp', 'ErrorSolution', 'SearchHistory']:
                    web_tools[tool_name] = description
                else:
                    exploration_tools[tool_name] = description
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🔧 Core Tools:**")
                for tool_name, description in core_tools.items():
                    st.write(f"• **{tool_name}**: {description}")
            
            with col2:
                st.write("**📊 Exploration Tools:**")
                for tool_name, description in exploration_tools.items():
                    st.write(f"• **{tool_name}**: {description}")
            
            with col3:
                st.write("**🔍 Web Tools:**")
                for tool_name, description in web_tools.items():
                    st.write(f"• **{tool_name}**: {description}")
    
    # Initialize chat history for ReAct agent
    if "react_chat_history" not in st.session_state:
        st.session_state.react_chat_history = []
    
    # Memory status và control buttons
    if 'react_agent_manager' in st.session_state:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            chat_stats = st.session_state.react_agent_manager.get_chat_stats()
            if chat_stats["memory_active"]:
                st.info(f"🧠 **Memory Active:** {chat_stats['context_messages']}/{chat_stats['max_context_messages']} messages in context (Total: {chat_stats['total_messages']})")
            else:
                st.info("🧠 **Memory:** No conversation history")
        
        with col2:
            max_messages = st.number_input("Max Context:", min_value=1, max_value=50, value=10, key="max_context_messages")
            if st.button("⚙️ Update Config"):
                st.session_state.react_agent_manager.set_max_context_messages(max_messages)
                st.success("✅ Config updated!")
        
        with col3:
            if st.button("🗑️ Clear Chat"):
                if st.session_state.react_agent_manager.clear_chat_history():
                    st.success("✅ Chat history cleared!")
                    st.rerun()
    
    # Display chat history
    for message in st.session_state.react_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Query composition helper with real-time token counting
    if 'react_agent_manager' in st.session_state:
        model_name = st.session_state.current_react_model
        token_manager = get_token_manager(model_name)
        
        with st.expander("✍️ Query Composer (với token counter)", expanded=False):
            # Text area for composing query
            query_text = st.text_area(
                "Soạn câu hỏi của bạn:",
                height=100,
                placeholder="Ví dụ: Hãy phân tích dữ liệu thiếu trong dataset và đưa ra khuyến nghị xử lý...",
                key="query_composer"
            )
            
            if query_text:
                # Real-time token counting
                query_info = token_manager.get_query_token_info(query_text)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens", query_info["tokens"])
                with col2:
                    st.metric("Limit", query_info["limit"])
                with col3:
                    color = "🔴" if query_info["is_over_limit"] else "🟢"
                    st.metric("Status", f"{color} {query_info['percentage']:.1f}%")
                
                # Progress bar
                progress_value = min(query_info["percentage"] / 100, 1.0)
                progress_color = "red" if query_info["is_over_limit"] else "green"
                st.progress(progress_value)
                
                # Warning if over limit
                if query_info["is_over_limit"]:
                    st.error(f"⚠️ Query vượt quá giới hạn {query_info['tokens'] - query_info['limit']} tokens. Sẽ được tự động rút gọn.")
                    
                    # Preview truncated version
                    truncated = token_manager.smart_truncate_query(query_text)
                    st.info("🔍 **Preview sau khi rút gọn:**")
                    st.write(truncated)
                
                # Send button
                if st.button("📤 Gửi Query", type="primary"):
                    st.session_state.composed_query = query_text
                    st.rerun()
    
    # Handle composed query
    if hasattr(st.session_state, 'composed_query') and st.session_state.composed_query:
        prompt = st.session_state.composed_query
        st.session_state.composed_query = None  # Clear after use
        
        # Add user message to chat history
        st.session_state.react_chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with ReAct agent
        if 'react_agent_manager' in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("ReAct Agent đang suy nghĩ..."):
                    try:
                        response = st.session_state.react_agent_manager.process_query(prompt)
                        st.write(response)
                        
                        # Add assistant response to chat history
                        st.session_state.react_chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ Lỗi ReAct agent: {str(e)}"
                        st.error(error_msg)
                        st.session_state.react_chat_history.append({"role": "assistant", "content": error_msg})
        else:
            st.error("❌ ReAct agent chưa được khởi tạo")
    
    # Chat input with length validation
    if prompt := st.chat_input("Hãy hỏi ReAct agent về dữ liệu của bạn..."):
        # Add user message to chat history
        st.session_state.react_chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with ReAct agent
        if 'react_agent_manager' in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("ReAct Agent đang suy nghĩ..."):
                    try:
                        response = st.session_state.react_agent_manager.process_query(prompt)
                        st.write(response)
                        
                        # Add assistant response to chat history
                        st.session_state.react_chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ Lỗi ReAct agent: {str(e)}"
                        st.error(error_msg)
                        st.session_state.react_chat_history.append({"role": "assistant", "content": error_msg})
        else:
            st.error("❌ ReAct agent chưa được khởi tạo")


def render_legacy_analysis_interface(declaration):
    """Render legacy analysis interface"""
    st.write("🔧 Legacy Analysis Interface")
    st.write(f"Available {declaration} analysis functions:")
    
    toolbar_buttons = get_analysis_toolbar(declaration)

    # Initialize session state for active function
    if 'active_analysis_function' not in st.session_state:
        st.session_state.active_analysis_function = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # If a function is active, show its input form
    if st.session_state.active_analysis_function:
        function_name = st.session_state.active_analysis_function
        analysis_tool = st.session_state.analysis_agent.tools[0]
        result = prompt_user_for_inputs(function_name, st.session_state.df, analysis_tool)
        if result is not None:
            # Log the function call and result to the console
            call_str = f"{analysis_tool.name}._run('{function_name}')"
            st.session_state.console_log.append({"call": call_str, "result": result})
            st.session_state.analysis_result = result
            st.session_state.active_analysis_function = None
            st.rerun()
        # Show result if available
        if st.session_state.analysis_result:
            # Check if result contains HTML image tag
            if "<img" in st.session_state.analysis_result:
                text_parts = st.session_state.analysis_result.split("<img")
                if text_parts[0].strip():
                    st.write(text_parts[0].strip())
                img_start = st.session_state.analysis_result.find("<img")
                img_end = st.session_state.analysis_result.find(">", img_start) + 1
                img_html = st.session_state.analysis_result[img_start:img_end]
                st.markdown(img_html, unsafe_allow_html=True)
                if len(text_parts) > 1:
                    after_img = text_parts[1].split(">", 1)
                    if len(after_img) > 1 and after_img[1].strip():
                        st.write(after_img[1].strip())
            else:
                # Check if the result is a DataFrame and clean it if needed
                if isinstance(st.session_state.analysis_result, pd.DataFrame):
                    clean_result = clean_dataframe_for_display(st.session_state.analysis_result)
                    st.dataframe(clean_result, use_container_width=True)
                else:
                    st.write(st.session_state.analysis_result)
            st.session_state.analysis_result = None
    else:
        # Show toolbar buttons if no function is active
        cols = st.columns(3)
        for i, (button_text, function_name) in enumerate(toolbar_buttons.items()):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(button_text, key=f"toolbar_{function_name}"):
                    st.session_state.active_analysis_function = function_name
                    st.rerun()

# Load environment variables
# Environment variables loaded in main app.py

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
    initialize_session_state()
    
    # Model selection and agent setup
    model_choice = render_model_selection()
    setup_agents(model_choice)
    setup_react_agent(model_choice)
    
    # Sidebar
    render_file_upload_sidebar()
    render_control_buttons()
    render_backup_status_indicator()
    # render_data_view_selector()  # Simplified: removed complex view switching
    
    # Main content area
    if st.session_state.df is not None:
        # --- Backup Notifications ---
        render_backup_notification()
        
        # --- Backup Management Section ---
        if st.sidebar.button("💾 Backup Management"):
            st.session_state.show_backup_panel = True
        
        if st.session_state.get('show_backup_panel', False):
            render_backup_control_panel()
            if st.button("❌ Close Backup Panel"):
                st.session_state.show_backup_panel = False
                st.rerun()
        
        # --- Data Quality Dashboard Section ---
        if st.sidebar.button("🔍 Data Quality Dashboard"):
            st.session_state.show_quality_dashboard = True
        
        if st.session_state.get('show_quality_dashboard', False):
            render_data_quality_dashboard()
            if st.button("❌ Close Quality Dashboard"):
                st.session_state.show_quality_dashboard = False
                st.rerun()
        
        # --- Processing Pipeline Section ---
        # render_processing_pipeline()  # Simplified: removed complex pipeline display
        
        # --- Data View Comparison Section ---
        # if st.sidebar.button("📊 Compare Data Versions"):
        #     st.session_state.show_data_comparison = True
        # 
        # if st.session_state.get('show_data_comparison', False):
        #     render_data_view_comparison()
        
        # --- Agent Data Source Warning ---
        # render_agent_data_source_warning()  # Simplified: removed complex warnings
        
        # --- Data Declaration Section ---
        declaration = render_data_declaration_section()

        # Handle specific data type declarations
        if declaration == "Panel Data":
            handle_panel_data_declaration(model_choice)
        elif declaration == "Time-Series":
            handle_time_series_declaration(model_choice)
        elif declaration == "Cross-Sectional":
            handle_cross_sectional_declaration(model_choice)

        # --- Analysis Toolbar ---
        if declaration != "None":
            st.subheader("🔧 Analysis Toolbar")
            st.write(f"Available {declaration} analysis functions:")
            
            toolbar_buttons = get_analysis_toolbar(declaration)

            # Initialize session state for active function
            if 'active_analysis_function' not in st.session_state:
                st.session_state.active_analysis_function = None
            if 'analysis_result' not in st.session_state:
                st.session_state.analysis_result = None

            # If a function is active, show its input form
            if st.session_state.active_analysis_function:
                function_name = st.session_state.active_analysis_function
                analysis_tool = st.session_state.analysis_agent.tools[0]
                result = prompt_user_for_inputs(function_name, st.session_state.df, analysis_tool)
                if result is not None:
                    # Log the function call and result to the console
                    call_str = f"{analysis_tool.name}._run('{function_name}')"
                    st.session_state.console_log.append({"call": call_str, "result": result})
                    st.session_state.analysis_result = result
                    st.session_state.active_analysis_function = None
                    st.rerun()
                # Show result if available
                if st.session_state.analysis_result:
                    # Check if result contains HTML image tag
                    if "<img" in st.session_state.analysis_result:
                        text_parts = st.session_state.analysis_result.split("<img")
                        if text_parts[0].strip():
                            st.write(text_parts[0].strip())
                        img_start = st.session_state.analysis_result.find("<img")
                        img_end = st.session_state.analysis_result.find(">", img_start) + 1
                        img_html = st.session_state.analysis_result[img_start:img_end]
                        st.markdown(img_html, unsafe_allow_html=True)
                        if len(text_parts) > 1:
                            after_img = text_parts[1].split(">", 1)
                            if len(after_img) > 1 and after_img[1].strip():
                                st.write(after_img[1].strip())
                    else:
                        # Check if the result is a DataFrame and clean it if needed
                        if isinstance(st.session_state.analysis_result, pd.DataFrame):
                            clean_result = clean_dataframe_for_display(st.session_state.analysis_result)
                            st.dataframe(clean_result, use_container_width=True)
                        else:
                            st.write(st.session_state.analysis_result)
                    st.session_state.analysis_result = None
            else:
                # Show toolbar buttons if no function is active
                cols = st.columns(3)
                for i, (button_text, function_name) in enumerate(toolbar_buttons.items()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if st.button(button_text, key=f"toolbar_{function_name}"):
                            st.session_state.active_analysis_function = function_name
                            st.rerun()

        # --- Analysis Console Section ---
        st.subheader("🖥️ Analysis Console")
        if st.button("Clear Console"):
            st.session_state.console_log = []
        for entry in reversed(st.session_state.console_log):
            st.code(entry["call"], language="python")
            if isinstance(entry["result"], pd.DataFrame):
                clean_result = clean_dataframe_for_display(entry["result"])
                st.dataframe(clean_result, use_container_width=True)
            else:
                st.write(entry["result"])
            st.markdown("---")
        
        # --- ReAct Agent Chat Interface ---
        render_react_chat_interface()
        
        # --- Legacy Analysis Interface ---
        if declaration != "None":
            with st.expander("🔧 Legacy Analysis Interface"):
                render_legacy_analysis_interface(declaration)
        
        # Data preview
        render_data_preview()
        
        # Display data types
        render_data_types_section(st.session_state.df)
        
        # Data Quality Checks section
        render_data_quality_checks(st.session_state.df)
    
    else:
        render_welcome_screen()

if __name__ == "__main__":
    main()