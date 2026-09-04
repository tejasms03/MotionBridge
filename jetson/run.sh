#!/usr/bin/env bash
# Run MotionBridge with Jetson config.
# Execute from anywhere:  bash jetson/run.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Verify ESP32 port
if [ ! -e /dev/ttyUSB0 ]; then
    echo "[WARNING] /dev/ttyUSB0 not found — ESP32 may not be connected."
    echo "          Check with: ls /dev/ttyUSB*"
fi

cd "$PROJECT_DIR"
python3 main.py --config jetson/config.yaml
