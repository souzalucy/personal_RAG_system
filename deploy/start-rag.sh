#!/bin/bash
# Start the Dual RAG System (API + Frontend)
# This script is called by the systemd service

cd /home/lucy/Pesquisa/RAG

# Kill any existing instances
pkill -f "uvicorn app.main:app" 2>/dev/null
pkill -f "streamlit run" 2>/dev/null
sleep 1

# Start API
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/rag-api.log 2>&1 &
API_PID=$!

# Wait for API to start
sleep 5

# Start Frontend
nohup streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/rag-frontend.log 2>&1 &
FRONTEND_PID=$!

echo "API PID: $API_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "API: http://localhost:8000"
echo "Frontend: http://localhost:8501"
