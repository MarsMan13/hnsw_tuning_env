#!/bin/bash

# This script copies all directories from a source directory (./result/{number})
# to a destination directory (./result/0).
# It takes one argument: the number for the source directory.

# --- 1. Check for Input Argument ---
# Check if a number was provided as an argument.
if [ -z "$1" ]; then
    echo "Error: No number provided."
    echo "Usage: ./copy_dirs.sh <number>"
    exit 1
fi

# --- 2. Define Source and Destination Directories ---
INPUT_NUMBER=$1
SOURCE_DIR="./result/$INPUT_NUMBER"
DEST_DIR="./result/0"

echo "Source directory: $SOURCE_DIR"
echo "Destination directory: $DEST_DIR"

# --- 3. Validate Directories ---
# Check if the source directory exists.
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create the destination directory if it doesn't exist.
# The '-p' flag prevents an error if the directory already exists.
mkdir -p "$DEST_DIR"
echo "Ensuring destination directory '$DEST_DIR' exists."
echo "---"

# --- 4. Find and Copy Directories ---
# Loop through all items in the source directory.
# The '*/' pattern ensures we only get directories.
for dir_path in "$SOURCE_DIR"/*/; do
    # Check if the found path is actually a directory to avoid errors with '*/'
    # if no directories are found.
    if [ -d "$dir_path" ]; then
        # Use 'basename' to get just the directory name for the log message.
        dir_name=$(basename "$dir_path")
        echo "Copying directory: '$dir_name' -> '$DEST_DIR'"
        
        # Use 'cp -r' to recursively copy the directory and its contents.
        # The quotes handle directory names with spaces.
        cp -r "$dir_path" "$DEST_DIR"
    fi
done

echo "---"
echo "All directories have been copied successfully."