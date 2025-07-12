import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from utils import clean_dataframe_for_display

from analysis_ui import get_analysis_toolbar
from analysis_forms import prompt_user_for_inputs
from data_declaration import (
    render_data_declaration_section,
    handle_panel_data_declaration,
    handle_time_series_declaration,
    handle_cross_sectional_declaration
)
from data_quality import render_data_types_section, render_data_quality_checks
from ui_components import (
    initialize_session_state,
    setup_agents,
    render_model_selection,
    render_file_upload_sidebar,
    render_control_buttons,
    render_chat_interface,
    render_data_preview,
    render_welcome_screen
)

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
    initialize_session_state()
    
    # Model selection and agent setup
    model_choice = render_model_selection()
    setup_agents(model_choice)
    
    # Sidebar
    render_file_upload_sidebar()
    render_control_buttons()
    
    # Main content area
    if st.session_state.df is not None:
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
        
        # Chat interface
        render_chat_interface(declaration)
        
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