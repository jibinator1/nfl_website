#!/bin/bash
# ==============================================================================
# NFL Analytics Hub - Mac / Linux Daily Update Runner
# ==============================================================================
# Run manually:
#     ./daily_update.sh
#
# Or add to crontab (e.g. daily at 6:00 AM):
#     0 6 * * * cd "/path/to/nfl_website" && ./daily_update.sh >> update.log 2>&1
# ==============================================================================

set -e

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "Starting Daily NFL Data Update: $(date)"
echo "Directory: $SCRIPT_DIR"
echo "=================================================================="

# Check for virtual environment in .venv or venv
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    echo "Activating virtual environment (venv)..."
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "Using system python..."
fi

# Run the python updater
python3 daily_update.py

echo ""
echo "Finished at: $(date)"
echo "=================================================================="
