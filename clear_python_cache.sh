#!/bin/bash
# clean_python_cache.sh
# This script removes all Python cache files from the project

echo "Cleaning __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Deleting .pyc files..."
find . -type f -name "*.pyc" -delete

echo "Deleting .pyo files (if any)..."
find . -type f -name "*.pyo" -delete

echo "Python cache cleaned!"
