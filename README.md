# 🤖 Data Preprocessing Chat MVP

An AI-powered Streamlit application that allows users to preprocess CSV/XLSX data through natural language conversations with an intelligent agent.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-v0.1+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 Features

- **🗣️ Natural Language Processing**: Chat with your data using plain English
- **📊 Smart Data Analysis**: AI-powered insights and recommendations
- **🔄 Interactive Workflow**: Apply/Undo changes with session management
- **📁 Multi-file Support**: Upload and merge multiple CSV/XLSX files
- **🎯 Structured Responses**: Professional analysis with actionable insights
- **⚡ Real-time Processing**: Instant data transformations and visualizations
- **🤖 Multiple AI Models**: Choose from GPT-4o, GPT-4-turbo, and more

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key
- Docker (optional, for future sandbox feature)

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/data-preprocessing-mvp.git](https://github.com/yourusername/data-preprocessing-mvp.git)
    cd data-preprocessing-mvp
    ```
2.  **Create virtual environment**
    ```bash
    python -m venv venv
    ```
    * **Windows**
        ```bash
        venv\Scripts\activate
        ```
    * **macOS/Linux**
        ```bash
        source venv/bin/activate
        ```
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Setup environment variables**
    ```bash
    cp .env.example .env
    # Edit .env and add your OpenAI API key
    ```
5.  **Run the application**
    ```bash
    streamlit run app.py
    ```
    Open in browser:
    * Local: `http://localhost:8501`
    * Network: `http://your-ip:8501`

---

## 💡 Usage Examples

### Basic Data Exploration

-   "Show me basic statistics of the data"
-   "What are the data types of all columns?"
-   "How many missing values are there?"

### Data Cleaning

-   "Remove rows with missing values"
-   "Fill missing values in 'age' column with the mean"
-   "Remove duplicate rows"

### Data Transformation

-   "Create a new column 'grade_category' based on scores"
-   "Convert 'date' column to datetime format"
-   "Normalize the 'price' column"

### Advanced Analysis

-   "Find outliers in 'salary' column using IQR method"
-   "Group by 'department' and calculate average salary"
-   "Create age groups and analyze performance by group"

### Multi-file Operations

-   "Merge all uploaded files on 'customer_id' column"
-   "Join sales data with customer data"
-   "Combine monthly reports into yearly summary"

---

## 📁 Project Structure

data-preprocessing-mvp/
├── app.py                   # Main Streamlit application
├── agent_manager.py         # AI agent management and prompting
├── utils.py                 # Utility functions and helpers
├── docker_sandbox.py        # Docker sandbox (future feature)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── examples/
│   └── sample_students.csv  # Sample data files
└── README.md                # Project documentation


---

## 🎛️ Configuration

### Model Selection

The app supports multiple OpenAI models:

-   `gpt-4o` (Recommended): Best balance of quality and speed
-   `gpt-4-turbo`: High quality, cost-effective
-   `gpt-4`: Reliable baseline
-   `o1-preview`: Complex reasoning tasks

### Environment Variables

```env
OPENAI_API_KEY=your_key_here     # Required
DEFAULT_MODEL=gpt-4o           # Optional