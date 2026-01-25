#!/bin/bash

# Usage:
#   ./copy_dirs.sh <src_number> [dst_number]
# If dst_number is omitted, it defaults to 0.

# --- 1. Check source argument ---
if [ -z "$1" ]; then
    echo "Error: No source number provided."
    echo "Usage: ./copy_dirs.sh <src_number> [dst_number]"
    exit 1
fi

SRC_NUM="$1"
DST_NUM="${2:-0}"   # default to 0 if not provided

SOURCE_DIR="./result/$SRC_NUM"
DEST_DIR="./result/$DST_NUM"

echo "Source directory      : $SOURCE_DIR"
echo "Destination directory : $DEST_DIR"
echo "---"

# --- 2. Validate source directory ---
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# --- 3. Ensure destination directory exists ---
mkdir -p "$DEST_DIR"

# --- 4. Copy subdirectories ---
for dir_path in "$SOURCE_DIR"/*/; do
    [ -d "$dir_path" ] || continue

    dir_name=$(basename "$dir_path")
    echo "Copying '$dir_name' -> '$DEST_DIR/'"
    cp -r "$dir_path" "$DEST_DIR"
done

echo "---"
echo "Done."
