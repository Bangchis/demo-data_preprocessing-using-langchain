#!/usr/bin/env python3
"""
Data Preprocessing Chat MVP - Main Application Entry Point

This file serves as the main entry point for the Streamlit application.
The actual application logic is organized in the src/ directory.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from config directory
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
if __name__ == "__main__":
    # Import module core_app (src/core/app.py)
    from src.core import app as core_app

    # Gọi hàm main() thực sự của ứng dụng
    core_app.main()