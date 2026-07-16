#!/usr/bin/env bash
# Removes macOS AppleDouble (._*) resource fork files and .DS_Store metadata files.
# Usage: bash cleanup_mac_artifacts.sh [--dry-run]

ROOT="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "--- DRY RUN: no files will be deleted ---"
fi

DS_STORE_FILES=$(find "$ROOT" -name ".DS_Store")
RESOURCE_FORKS=$(find "$ROOT" -name "._*")

DS_COUNT=$(echo "$DS_STORE_FILES" | grep -c . || true)
RF_COUNT=$(echo "$RESOURCE_FORKS" | grep -c . || true)

echo "Found $DS_COUNT .DS_Store files"
echo "Found $RF_COUNT ._* resource fork files"

if $DRY_RUN; then
    echo ""
    echo ".DS_Store files:"
    echo "$DS_STORE_FILES"
    echo ""
    echo "._* resource fork files (first 20):"
    echo "$RESOURCE_FORKS" | head -20
    echo "..."
    exit 0
fi

echo "Deleting..."

find "$ROOT" -name ".DS_Store" -delete
find "$ROOT" -name "._*" -delete

echo "Done. Deleted $DS_COUNT .DS_Store files and $RF_COUNT ._* resource fork files."
