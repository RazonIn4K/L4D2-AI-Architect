#!/bin/bash
# L4D2 SourcePawn Code Generator - Web UI Launcher
#
# Usage:
#   ./start_web_ui.sh              # Start on default port 8000
#   ./start_web_ui.sh 8080         # Start on custom port
#   ./start_web_ui.sh --public     # Allow network access (0.0.0.0)
#   ./start_web_ui.sh --dashboard  # Start the model comparison dashboard
#   ./start_web_ui.sh --dashboard 8501  # Dashboard on custom port

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for dashboard mode
DASHBOARD_MODE=false
if [[ "$1" == "--dashboard" ]] || [[ "$2" == "--dashboard" ]]; then
    DASHBOARD_MODE=true
fi

# Default ports
if [ "$DASHBOARD_MODE" = true ]; then
    PORT=${2:-8501}
    if [[ "$1" == "--dashboard" ]]; then
        PORT=${2:-8501}
    fi
else
    PORT=${1:-8000}
fi

HOST="127.0.0.1"

# Check for --public flag
if [[ "$1" == "--public" ]] || [[ "$2" == "--public" ]] || [[ "$3" == "--public" ]]; then
    HOST="0.0.0.0"
    if [[ "$1" == "--public" ]]; then
        PORT=${2:-8000}
    fi
fi

if [ "$DASHBOARD_MODE" = true ]; then
    echo "=============================================="
    echo "  L4D2 Model Comparison Dashboard"
    echo "=============================================="
else
    echo "=============================================="
    echo "  L4D2 SourcePawn Code Generator - Web UI"
    echo "=============================================="
fi
echo

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    # Try loading from .env
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | grep OPENAI_API_KEY | xargs)
    fi

    if [ -z "$OPENAI_API_KEY" ]; then
        echo "WARNING: OPENAI_API_KEY not set"
        echo
        echo "Set it with one of these methods:"
        echo "  1. export OPENAI_API_KEY='sk-...'"
        echo "  2. Add OPENAI_API_KEY=sk-... to .env file"
        echo "  3. Use Doppler: doppler run -- ./start_web_ui.sh"
        echo
    fi
fi

# Check for Python and dependencies
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

if [ "$DASHBOARD_MODE" = true ]; then
    # Check for Streamlit dependencies
    python3 -c "import streamlit" 2>/dev/null || {
        echo "Installing required packages..."
        pip install streamlit pandas plotly
    }

    echo "Starting Model Comparison Dashboard..."
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  URL:  http://${HOST}:${PORT}"
    echo
    echo "Press Ctrl+C to stop"
    echo "----------------------------------------------"
    echo

    if [[ "$HOST" == "0.0.0.0" ]]; then
        streamlit run scripts/evaluation/model_dashboard.py --server.port "$PORT" --server.address "$HOST"
    else
        streamlit run scripts/evaluation/model_dashboard.py --server.port "$PORT"
    fi
else
    # Check for required packages
    python3 -c "import fastapi, uvicorn, openai" 2>/dev/null || {
        echo "Installing required packages..."
        pip install fastapi uvicorn openai python-dotenv
    }

    echo "Starting Web UI..."
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  URL:  http://${HOST}:${PORT}"
    echo
    echo "Press Ctrl+C to stop"
    echo "----------------------------------------------"
    echo

    python3 scripts/inference/web_ui.py --host "$HOST" --port "$PORT"
fi
