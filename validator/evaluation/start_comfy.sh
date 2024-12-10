#!/bin/bash

COMFYUI_DIR="ComfyUI"
VENV_DIR="$COMFYUI_DIR/venv"
MAIN_SCRIPT="$COMFYUI_DIR/main.py"
LOG_FILE="$COMFYUI_DIR/comfyui.log"
PID_FILE="$COMFYUI_DIR/comfyui.pid"

start_comfyui() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "ComfyUI is already running with PID $(cat $PID_FILE)."
        exit 0
    fi

    echo "Starting ComfyUI..."
    nohup "$VENV_DIR/bin/python" "$MAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "ComfyUI started with PID $!"
}

stop_comfyui() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "Stopping ComfyUI with PID $(cat $PID_FILE)..."
        kill "$(cat $PID_FILE)" && rm -f "$PID_FILE"
        echo "ComfyUI stopped."
    else
        echo "ComfyUI is not running."
    fi
}

case "$1" in
    start)
        start_comfyui
        ;;
    stop)
        stop_comfyui
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
