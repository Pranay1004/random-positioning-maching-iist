#!/bin/bash
# RPM Digital Twin Streamlit App Launcher
# Usage: ./run_streamlit.sh

echo "🌐 RPM Digital Twin - Streamlit Launcher"
echo "========================================"

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment  
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "✅ Virtual environment activated"

# Check if requirements are installed
python -c "import streamlit, numpy, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Streamlit dependencies..."
    pip install -r requirements-streamlit.txt
fi

echo "🚀 Starting RPM Digital Twin on Streamlit..."
echo ""
echo "🔗 Open: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Launch Streamlit app
streamlit run rpm_streamlit_app.py --server.port 8501 --browser.gatherUsageStats false