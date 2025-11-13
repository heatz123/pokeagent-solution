#!/bin/bash
# Code synchronization script - syncs Python and HTML files between two directories
# Usage: ./sync_code.sh [destination_path]

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Source directory (current directory)
SOURCE_DIR="$(pwd)"

# Destination directory (passed as argument or default)
DEST_DIR="${1:-}"

# Default destination if not specified
if [ -z "$DEST_DIR" ]; then
    echo -e "${YELLOW}Usage: $0 <destination_directory>${NC}"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/other/pokeagent-folder"
    echo ""
    echo "This will sync all .py and .html files from current directory to destination"
    exit 1
fi

# Check if destination exists
if [ ! -d "$DEST_DIR" ]; then
    echo -e "${YELLOW}Warning: Destination directory does not exist: $DEST_DIR${NC}"
    read -p "Create it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$DEST_DIR"
    else
        exit 1
    fi
fi

echo -e "${BLUE}🔄 Syncing Python and HTML files...${NC}"
echo -e "   Source: ${GREEN}$SOURCE_DIR${NC}"
echo -e "   Dest:   ${GREEN}$DEST_DIR${NC}"
echo ""

# Rsync options:
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -u: skip files that are newer on the destination
# --include: include pattern
# --exclude: exclude pattern
# --delete: delete files in dest that don't exist in source (optional, commented out by default)

rsync -avu \
    --include='*/' \
    --include='*.py' \
    --include='*.html' \
    --exclude='*' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pokeagent_cache/' \
    --exclude='venv/' \
    --exclude='env/' \
    --exclude='.venv/' \
    "$SOURCE_DIR/" "$DEST_DIR/"

# Optional: Add --delete flag to remove files in dest that don't exist in source
# Uncomment the line below if you want this behavior:
# Add --delete to the rsync command above

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Sync completed successfully!${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Sync completed with errors (exit code: $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE
