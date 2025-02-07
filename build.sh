#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p models uploads backups logs static tmp

# Initialize data files if they don't exist
if [ ! -f diseases_data.json ]; then
    echo "{}" > diseases_data.json
fi

# Set proper permissions
chmod -R 755 models uploads backups logs static tmp
chmod 644 diseases_data.json

# Clear any temp files
rm -f tmp/*