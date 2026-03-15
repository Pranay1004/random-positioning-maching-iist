#!/bin/bash
source /Users/pandeyji/Desktop/RPM/programming/.venv/bin/activate
cd /Users/pandeyji/Desktop/RPM/programming/Digital_Twin
PORT=8080
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do PORT=$((PORT+1)); done
PYTHONPATH=. python -m uvicorn src.webapp.server:app --host 0.0.0.0 --port $PORT --reload

