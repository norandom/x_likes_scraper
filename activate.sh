#!/bin/bash
# Activation script for X Likes Exporter virtual environment

echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated!"
echo ""
echo "You can now run:"
echo "  python cli.py cookies.json YOUR_USER_ID"
echo ""
echo "To deactivate when done, type: deactivate"
