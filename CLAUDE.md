# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Data Preprocessing Chat MVP** - a sophisticated Streamlit application that enables users to preprocess CSV/XLSX data through natural language conversations with AI agents. The application features a modern architecture with dual AI systems:

1. **ReAct Agent** (`src/agents/react_agent.py`): Advanced intelligent agent with 16 specialized tools for data exploration, code execution, and web search
2. **Legacy Analysis Agent** (`src/agents/analysis_agent.py`): Structured analysis system for Panel Data, Time-Series, and Cross-Sectional data

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

### Directory Structure
```
src/
├── core/
│   ├── app.py               # Main Streamlit application logic
│   ├── utils.py             # Data loading and DataFrame utilities
│   └── backup_manager.py    # Comprehensive backup system
├── agents/
│   ├── react_agent.py       # ReAct agent with 16 tools
│   └── analysis_agent.py    # Legacy structured analysis agent
├── tools/
│   ├── core.py              # Core execution tools (CodeRunner, Undo, Redo)
│   ├── basic.py             # Data exploration tools (QuickInfo, MissingReport)
│   ├── web.py               # Web search tools (WebSearch, PandasHelp)
│   ├── backup_tools.py      # Backup-related tools
│   └── enhanced_structural_detection.py  # Advanced structural error detection
├── analysis/
│   ├── declaration.py       # Data type declaration UI
│   ├── quality.py           # Data quality checks
│   ├── forms.py             # Analysis input forms
│   ├── ui.py                # Analysis toolbar UI
│   └── utils/               # Analysis utilities by data type
├── ui/
│   ├── components.py        # Reusable UI components
│   └── backup_components.py # Backup system UI components
└── utils/
    └── token_manager.py     # Token usage tracking and management
```

### Key Architectural Patterns

#### Dual DataFrame System
The application maintains two synchronized DataFrame versions:
- **`st.session_state.df_original`**: Preserves original data types (datetime, numeric, boolean) for analysis
- **`st.session_state.df`**: Streamlit-compatible display version (converted to strings) for UI rendering

#### ReAct Agent Tool System
The agent uses 16 specialized tools organized in three categories:

**Core Tools (4)** - Execution and state management:
- `CodeRunner`: Execute pandas code safely in sandboxed environment
- `Undo`: Revert to previous DataFrame state
- `Redo`: Reapply previously undone operation
- `ExecutionLog`: View recent code execution history

**Basic Tools (9)** - Data exploration and analysis:
- `QuickInfo`: Quick DataFrame overview with structural error detection
- `FullInfo`: Comprehensive data report (schema, missing, duplicates, outliers)
- `MissingReport`: Detailed missing value analysis
- `DuplicateCheck`: Identify duplicate rows
- `ColumnSummary`: Categorize columns by data type
- `ValueCounts`: Top value frequencies for specific columns
- `CorrelationMatrix`: High correlation pairs identification
- `OutlierCheck`: Outlier detection using IQR method
- `SchemaReport`: Data types and missing value summary

**Web Tools (5)** - External knowledge and help:
- `WebSearch`: Real-time web search using Tavily API
- `PandasHelp`: Pandas-specific help from official docs and Stack Overflow
- `DataScienceHelp`: Data science guidance from Kaggle and Medium
- `ErrorSolution`: Error-specific solutions from Stack Overflow and GitHub
- `SearchHistory`: View previous search queries

**Backup Tools (4)** - Data protection and recovery:
- `CreateBackup`: Create manual backups with custom names
- `RestoreBackup`: Restore from existing backups
- `ListBackups`: View all available backups
- `DeleteBackup`: Remove outdated backups

## Common Development Commands

### Application Startup
```bash
# Start the main application
streamlit run app.py

# Start with specific configuration
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Start with auto-reload for development
streamlit run app.py --server.runOnSave true
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables in .env file at root
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

# Alternative: set environment variables directly
export OPENAI_API_KEY=your_key
export TAVILY_API_KEY=your_key
```

### Testing Commands
```bash
# Run unit tests
python tests/unit/test_data_integrity.py
python tests/unit/test_session_state.py
python tests/unit/test_chat_memory.py

# Run integration tests
python tests/integration/test_web_search_tools.py
python tests/integration/test_final_web_tools.py
python tests/integration/test_query_length_integration.py

# Run demo tests for development
python tests/demo/test_tavily_search.py
python tests/demo/test_simple_search.py
python tests/demo/test_tavily_direct_api.py

# Run system tests
python tests/test_backup_system.py
python tests/test_enhanced_structural_detection.py
python tests/test_token_management.py

# Run specific demonstrations
python demo_chat_memory.py
python demo_query_length.py
python final_answer_demo.py
```

### Development Workflow
```bash
# Check application structure
ls -la src/
ls -la tests/

# View backup system status
ls -la backups/

# Development mode with debugging
streamlit run app.py --logger.level debug

# Clear Streamlit cache during development
streamlit cache clear
```

## Session State Management

### Critical Session State Variables
```python
# DataFrames
st.session_state.df                 # Main display DataFrame
st.session_state.df_original        # Original typed DataFrame
st.session_state.dfs                # Multiple DataFrames dictionary

# Chat Systems
st.session_state.react_chat_history # ReAct agent conversation
st.session_state.chat_history       # Legacy analysis chat

# State Management
st.session_state.checkpoints        # Undo/redo checkpoints
st.session_state.checkpoint_index   # Current checkpoint position
st.session_state.execution_log      # Code execution history
st.session_state.web_search_log     # Web search history

# UI State
st.session_state.active_analysis_function  # Current analysis function
st.session_state.console_log              # Analysis console output
```

### DataFrame Type Preservation

The application implements intelligent type inference:
```python
# Key functions in src/core/utils.py
_apply_intelligent_type_inference(df)  # Convert Excel strings to proper types
_create_display_version(df)            # Create Streamlit-safe display version
clean_dataframe_for_display(df)        # Handle inf/nan values for display
```

## ReAct Agent System

### Architecture
The ReAct agent (`src/agents/react_agent.py`) implements a sophisticated conversation system:

1. **Context Management**: Maintains conversation history with configurable message limits
2. **Tool Integration**: Seamlessly integrates 16 specialized tools
3. **Safety System**: Code execution sandbox with dangerous operation blocking
4. **Memory System**: Persistent conversation context across sessions

### Agent Configuration
```python
# Agent initialization in src/agents/react_agent.py
max_context_messages = 10    # Configurable context window
max_iterations = 9           # Maximum ReAct iterations
early_stopping_method = "generate"
handle_parsing_errors = True
```

### Prompt Engineering
The agent uses Vietnamese-language prompts with structured ReAct format:
```
Thought: [Analysis and planning]
Action: [Tool name]
Action Input: [Specific parameters]
Observation: [Tool results]
```

## Data Type Declaration System

The application supports three data analysis paradigms:

### Panel Data
- **Variables**: Panel variable (entity ID) + Time variable
- **Analysis Functions**: `xtdescribe`, `xtsum`, `xttab`, `xtline`, `xtreg`, `xtlogit`, `xtprobit`, `xtpoisson`, `xtunitroot`
- **Use Cases**: Entity-time observations, fixed effects models

### Time-Series
- **Variables**: Time variable (sequential temporal data)
- **Analysis Functions**: `tsset`, `arima`, `dfuller`, `corrgram`, `var_model`, `vargranger`, `tsline`, `newey`
- **Use Cases**: Temporal analysis, forecasting, unit root tests

### Cross-Sectional
- **Variables**: No special requirements
- **Analysis Functions**: `summarize`, `tabulate`, `correlate`, `regress`, `logit`, `probit`, `ttest`, `chi2`, `histogram`, `scatter`
- **Use Cases**: Single time point analysis, correlation, regression

## Code Safety and Execution

### Safety Measures (`src/tools/core.py`)
```python
# Blocked operations for security
dangerous_patterns = [
    'import os', 'import sys', 'import subprocess',
    'exec(', 'eval(', 'open(', '__import__',
    'globals(', 'locals(', 'getattr(', 'setattr('
]
```

### Checkpoint System
- **Automatic Checkpoints**: Saved before each code execution
- **Undo/Redo**: Navigate through DataFrame state history
- **Memory Management**: Configurable history limit (default: 10 operations)

### Code Execution Flow
1. **Input Cleaning**: Remove comments, quotes, and markdown formatting
2. **Safety Check**: Scan for dangerous operations
3. **Checkpoint**: Save current DataFrame state
4. **Execution**: Run code in isolated environment
5. **Result Handling**: Update session state and provide feedback

## File Upload and Data Processing

### Supported Formats
- CSV files with various encodings
- Excel files (.xlsx) with proper type inference
- Multiple file uploads with automatic merging

### Type Inference Features
- **Numeric Detection**: Handles Excel numeric strings
- **DateTime Detection**: Recognizes various date formats
- **Boolean Detection**: Converts TRUE/FALSE, True/False to boolean
- **Error Handling**: Graceful fallback for problematic data

### Data Quality Checks
- **Structural Errors**: Detects malformed rows with delimiter issues
- **Missing Values**: Comprehensive missing data analysis
- **Duplicates**: Identifies and reports duplicate records
- **Outliers**: Statistical outlier detection using IQR method

## Web Search Integration

### Tavily API Integration
- **Real-time Search**: Current information via Tavily API
- **Domain Filtering**: Targeted searches (pandas.pydata.org, stackoverflow.com)
- **Query Logging**: Persistent search history
- **Rate Limiting**: Handles API limits gracefully

### Search Query Patterns
```python
# Simple search
"pandas pivot table"

# Domain-specific search
"pandas pivot table | domains=stackoverflow.com,pandas.pydata.org"

# Error-specific search
"pandas KeyError: 'column_name'"
```

## Development Guidelines

### Code Organization
- **Modular Design**: Clear separation between UI, agents, tools, and analysis
- **Type Hints**: Use typing annotations for better code clarity
- **Error Handling**: Comprehensive try-catch blocks with user-friendly messages
- **Logging**: Detailed execution and search logging

### Adding New Tools
1. **Create Tool Function**: In appropriate tools/ file
2. **Add to Agent**: Register in `src/agents/react_agent.py`
3. **Update Documentation**: Add to tool descriptions
4. **Test Integration**: Verify tool works with ReAct agent

### UI Development
- **Component Reuse**: Leverage `src/ui/components.py`
- **Session State**: Maintain proper state management
- **Vietnamese Support**: Use Vietnamese language for user-facing text
- **Error Messages**: Provide clear, actionable error messages

## Model Configuration

### Supported Models
- **gpt-4o**: Recommended for balanced performance and cost
- **gpt-4-turbo**: High quality, cost-effective option
- **gpt-4**: Reliable baseline model
- **o1-preview**: For complex reasoning tasks

### Model Selection
```python
# In UI model selector
model_options = {
    "gpt-4o": "GPT-4 Omni (Recommended)",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4": "GPT-4",
    "o1-preview": "O1 Preview"
}
```

## Troubleshooting Common Issues

### DataFrame Type Issues
- **Problem**: Mathematical operations fail on string columns
- **Solution**: Use `df_original` for analysis, `df` for display

### Code Execution Errors
- **Problem**: Agent code contains comments or formatting
- **Solution**: Code cleaning function removes comments automatically

### Session State Corruption
- **Problem**: Inconsistent session state across reloads
- **Solution**: Use `initialize_session_state()` function

### Web Search Failures
- **Problem**: Tavily API limits exceeded
- **Solution**: Implement graceful fallback to cached results

## Extension Points

### Adding New Analysis Types
1. **Create Utility Module**: In `src/analysis/utils/`
2. **Add Declaration Handler**: In `src/analysis/declaration.py`
3. **Update Agent**: Add analysis functions to `src/agents/analysis_agent.py`
4. **UI Integration**: Add to analysis toolbar

### Custom Tool Development
1. **Tool Function**: Create in appropriate `src/tools/` file
2. **LangChain Integration**: Use Tool class wrapper
3. **Agent Registration**: Add to tool list in `src/agents/react_agent.py`
4. **Documentation**: Update tool descriptions

### UI Enhancements
1. **Component Creation**: Add to `src/ui/components.py`
2. **State Management**: Ensure proper session state handling
3. **Language Support**: Maintain Vietnamese language consistency
4. **Error Handling**: Provide user-friendly error messages

## Backup System

### Architecture
The backup system (`src/core/backup_manager.py`) provides comprehensive data protection:

1. **Automatic Backups**: Created before risky operations
2. **Manual Backups**: User-initiated with custom names and descriptions
3. **Point-in-Time Recovery**: Restore to any previous state
4. **Persistent Storage**: Compressed pickle files with metadata

### Backup Directory Structure
```
backups/
├── backup_metadata.json     # Backup catalog and metadata
├── sessions/                # Session state backups
├── dataframes/              # DataFrame backups (compressed)
└── checkpoints/             # Checkpoint backups
```

### Key Functions
```python
# In src/core/backup_manager.py
create_backup(name, description)           # Create manual backup
restore_backup(backup_id)                  # Restore from backup
list_backups()                             # Get all backups
delete_backup(backup_id)                   # Remove backup
auto_backup(operation_type)                # Automatic backup
```

### Backup UI Components
- **Backup Control Panel**: Full backup management interface
- **Backup Status Indicator**: Shows backup system status
- **Backup Notifications**: User feedback for backup operations
- **Quick Backup**: One-click backup creation

## Token Management

### Token Tracking System
The application includes comprehensive token usage tracking (`src/utils/token_manager.py`):

1. **Per-Session Tracking**: Monitor token usage per session
2. **Model-Specific Metrics**: Track different model usage patterns
3. **Cost Estimation**: Estimate costs based on token usage
4. **Memory Management**: Optimize context window based on token limits

### Token Manager Functions
```python
# In src/utils/token_manager.py
get_token_manager()                        # Get singleton token manager
track_conversation_tokens(messages)        # Track tokens in conversation
get_session_stats()                        # Get current session statistics
optimize_context_window(messages)          # Optimize based on token limits
```

### Token Usage Monitoring
- **Real-time Display**: Shows current session token usage
- **Model Comparison**: Compare usage across different models
- **Context Optimization**: Automatically manage conversation context
- **Cost Tracking**: Monitor API costs based on token usage

## Enhanced Structural Detection

### Advanced Error Detection
The system includes sophisticated structural error detection (`src/tools/enhanced_structural_detection.py`):

1. **Delimiter Issues**: Detects CSV parsing problems
2. **Malformed Rows**: Identifies rows with incorrect column counts
3. **Encoding Problems**: Handles various character encodings
4. **Data Type Inconsistencies**: Detects mixed data types in columns

### Detection Functions
```python
# In src/tools/enhanced_structural_detection.py
detect_structural_errors(df)              # Comprehensive error detection
analyze_delimiter_issues(file_path)       # CSV delimiter analysis
detect_encoding_issues(file_path)         # Character encoding detection
validate_data_consistency(df)             # Data type consistency checks
```

## Important Notes

- **API Keys**: Both OpenAI and Tavily API keys are required
- **Language**: Vietnamese language used in prompts and UI
- **Docker**: Sandbox support exists but is currently disabled
- **Memory**: Configurable context window for conversation history
- **Safety**: Comprehensive code execution safety measures
- **Logging**: Detailed execution and search activity logging
- **Backup System**: Automatic and manual backup creation with recovery
- **Token Management**: Comprehensive usage tracking and optimization
- **Enhanced Detection**: Advanced structural error detection and handling