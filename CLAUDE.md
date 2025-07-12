# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Overview

This is a **Data Preprocessing MVP** - an AI-powered Streamlit application that provides conversational data analysis with comprehensive statistical capabilities. The application combines traditional econometric methods (STATA-inspired) with modern AI conversational interfaces for data preprocessing, analysis, and visualization.

## Common Development Commands

### Running the Application
```bash
streamlit run app.py
```
The app runs on `http://localhost:8501`

### Environment Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables
- Create `.env` file with `OPENAI_API_KEY=your_key_here`
- Optional: `DEFAULT_MODEL=gpt-4o`

## Code Architecture

### Core Application Flow
1. **`app.py`** - Main Streamlit entry point, orchestrates UI and workflow
2. **`utils.py`** - Core utilities for data loading, cleaning, session management
3. **AI Agent Layer**:
   - `agent_manager.py` - General data preprocessing agent using LangChain
   - `analysis_agent.py` - Specialized statistical analysis agent with custom tools

### UI Component System
- **`ui_components.py`** - Modular UI components (file upload, chat interface, welcome screen)
- **`analysis_ui.py`** - Analysis-specific UI components and toolbars
- **`analysis_forms.py`** - Dynamic form generation for statistical functions

### Data Management Layer
- **`data_declaration.py`** - Data type declaration system (Panel/Time-Series/Cross-Sectional)
- **`data_quality.py`** - Comprehensive data quality checks and STATA-style summaries

### Statistical Analysis Modules
- **`panel_utils.py`** - Panel data analysis (xtreg, xtsum, xtline, etc.)
- **`time_series_utils.py`** - Time series analysis (ARIMA, VAR, unit root tests)
- **`cross_sectional_utils.py`** - Cross-sectional analysis (regression, correlation)

## Key Technology Stack

- **Frontend**: Streamlit 1.46.1 with matplotlib/seaborn for visualization
- **AI Integration**: LangChain 0.3.26 + OpenAI 1.93.0 for conversational agents
- **Data Processing**: Pandas 2.3.0, NumPy 2.3.1, PyArrow 20.0.0
- **Statistical Analysis**: StatsModels 0.14.5, SciPy 1.16.0, LinearModels 6.1
- **File Support**: OpenPyXL 3.1.5 for Excel files

## Data Type System

The application adapts its UI and available functions based on declared data types:

1. **Panel Data**: Entity-time structure with panel-specific econometric tools
2. **Time Series**: Temporal data with ARIMA, VAR, stationarity tests
3. **Cross-Sectional**: Single-point-in-time data with regression and correlation analysis

Each data type triggers different UI toolbars and available analysis functions.

## Session State Management

The application maintains comprehensive session state:
- `st.session_state.data` - Current DataFrame
- `st.session_state.data_history` - Undo/redo functionality
- `st.session_state.chat_history` - Conversation memory
- `st.session_state.data_type` - Declared data structure type

## AI Agent Architecture

### General Agent (`agent_manager.py`)
- Uses LangChain's pandas DataFrame agent
- Handles data preprocessing, cleaning, transformation tasks
- Provides conversational interface for general data operations

### Analysis Agent (`analysis_agent.py`)
- Custom tool-based agent for statistical analysis
- Maps natural language to specific statistical functions
- Supports parameter extraction and function execution
- Provides educational explanations with results

## File Structure Conventions

- Main application logic in root directory
- UI components are modular and reusable
- Statistical utilities are organized by data type
- Agent files handle AI-powered functionality
- Session state management in `utils.py`

## Development Notes

- The application follows STATA naming conventions for statistical functions
- All statistical outputs include educational explanations
- Error handling includes user-friendly messages and suggestions
- The codebase supports both programmatic and conversational data analysis workflows