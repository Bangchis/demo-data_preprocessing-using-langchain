# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Preprocessing Chat MVP** - a Streamlit application that enables users to preprocess CSV/XLSX data through natural language conversations with AI agents. The application features two main AI systems:

1. **ReAct Agent**: Intelligent agent with web search capabilities and specialized tools
2. **Legacy Analysis Agent**: Data-type-specific analysis functions for Panel Data, Time-Series, and Cross-Sectional data

## Architecture

The application follows a modular architecture with clear separation of concerns:

### Core Components

- **`app.py`**: Main Streamlit application entry point with UI orchestration
- **`agent_manager.py`**: Manages the ReAct agent with LangChain integration
- **`ui_components.py`**: Reusable UI components and session state management
- **`utils.py`**: Data loading, type preservation, and DataFrame utilities

### Data Analysis System

- **`analysis_agent.py`**: Legacy analysis agent for structured data analysis
- **`data_declaration.py`**: Data type declaration UI (Panel/Time-Series/Cross-Sectional)
- **`data_quality.py`**: Data quality checks and validation
- **`*_utils.py`**: Specialized utilities for different data types (panel, time-series, cross-sectional)

### Tool System

The ReAct agent uses three categories of tools:

- **`tools_core.py`**: Core execution tools (CodeRunner, Undo/Redo, ExecutionLog)
- **`tools_basic.py`**: Data exploration tools (QuickInfo, MissingReport, etc.)
- **`tools_web.py`**: Web search and help tools (WebSearch, PandasHelp, etc.)

## Common Development Commands

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8501
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# OPENAI_API_KEY=your_key_here
```

### Testing

```bash
# Run test files (if any)
python test_data_integrity.py
python test_session_state.py
```

## Key Features

### Dual DataFrame System

The application maintains two versions of DataFrames:
- **`df_original`**: Preserves original data types (datetime, numeric, boolean)
- **`df_display`**: Streamlit-compatible version (strings) for UI display

### Session State Management

Critical session state variables:
- `df`, `dfs`: Current and multiple DataFrames
- `chat_history`, `react_chat_history`: Chat conversations
- `checkpoints`, `checkpoint_index`: Undo/redo functionality
- `execution_log`: Code execution history
- `web_search_log`: Web search history

### Data Type Handling

The application supports intelligent data type inference:
- Auto-detects numeric, datetime, and boolean columns
- Preserves original types for analysis
- Converts to display-friendly formats for UI

## ReAct Agent System

The ReAct agent (`agent_manager.py`) uses a structured approach:

1. **Tools**: 16 specialized tools across 3 categories
2. **Prompting**: Vietnamese language prompts with ReAct format
3. **Safety**: Code execution sandbox with safety checks
4. **Logging**: Comprehensive execution and search logging

### Tool Categories

**Core Tools**: CodeRunner, Undo, Redo, ExecutionLog
**Exploration Tools**: QuickInfo, MissingReport, DuplicateCheck, etc.
**Web Tools**: WebSearch, PandasHelp, DataScienceHelp, etc.

## Code Safety

The `tools_core.py` implements safety measures:
- Blocks dangerous operations (file system, imports, etc.)
- Sandboxed execution environment
- Checkpoint system for undo/redo
- Error handling and logging

## Data Declaration System

The application supports three data types:
- **Panel Data**: Entity-time observations
- **Time-Series**: Sequential temporal data
- **Cross-Sectional**: Single time point observations

Each type has specialized analysis functions and UI components.

## Important Notes

- The application uses OpenAI models (gpt-4o, gpt-4-turbo, etc.)
- Docker sandbox support exists but is currently disabled
- Vietnamese language support in prompts and UI
- Comprehensive error handling and user feedback
- Type-preserving data loading with fallback mechanisms

## File Upload System

The application supports:
- CSV and XLSX file formats
- Multiple file uploads
- Type-preserving data loading
- Intelligent data type inference
- Excel-specific handling (TRUE/FALSE, dates, etc.)

## Development Tips

- Use `load_uploaded_file_preserve_types()` for new file loading
- Maintain both original and display DataFrames
- Test with various data types and edge cases
- Check session state management for new features
- Use the tool system for extending functionality