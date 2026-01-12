#!/bin/bash

BASE_URL="http://localhost:8000"

case "$1" in
    start-app)
        echo "Starting FastAPI app..."
        nohup python3 app.py > scanner_app.log 2>&1 &
        echo "App started in background (PID: $!)."
        ;;
    stop-app)
        echo "Stopping FastAPI app..."
        pkill -f "python3 app.py"
        echo "App stopped."
        ;;
    start-scan)
        echo "Starting scanner loop..."
        curl -X POST "$BASE_URL/scan/start"
        echo -e "\nDone."
        ;;
    stop-scan)
        echo "Stopping scanner loop..."
        curl -X POST "$BASE_URL/scan/stop"
        echo -e "\nDone."
        ;;
    status)
        curl -s "$BASE_URL/healthz" | python3 -m json.tool
        ;;
    report)
        curl -s "$BASE_URL/report/insiders" | python3 -m json.tool
        ;;
    *)
        echo "Usage: $0 {start-app|stop-app|start-scan|stop-scan|status|report}"
        exit 1
esac
