# utils.py
"""Utility functions for the application"""

import os
from pathlib import Path
from typing import Optional
import streamlit as st


def load_api_key() -> Optional[str]:
    """Load OpenAI API key from environment or Streamlit secrets"""
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    # Try Streamlit secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    return None


def setup_environment():
    """Setup environment and check requirements"""
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        print("Created .env file. Please add your OpenAI API key.")
    
    # Load from .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env")
    except ImportError:
        print("python-dotenv not installed. Install with: pip install python-dotenv")


def format_time_ms(time_seconds: float) -> str:
    """Format time in milliseconds"""
    return f"{time_seconds * 1000:.0f}ms"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def get_status_color(value: float, thresholds: dict) -> str:
    """Get color based on value and thresholds"""
    if value >= thresholds.get('good', 0.7):
        return "🟢"
    elif value >= thresholds.get('ok', 0.5):
        return "🟡"
    else:
        return "🔴"