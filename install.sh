#!/bin/bash
set -e

# Configuration
BINARY_NAME="oxidane"
GITHUB_REPO="um-univie/oxidane"
INSTALL_DIR="/usr/local/bin"

# Parse arguments
VERSION=${1:-latest}

# Determine OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
case $ARCH in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    armv7l)
        ARCH="armv7"
        ;;
esac

# Determine file extension
if [ "$OS" = "windows" ]; then
    EXT=".exe"
else
    EXT=""
fi

echo "Detected OS: $OS, Architecture: $ARCH"

# Construct download URL
if [ "$VERSION" = "latest" ]; then
    DOWNLOAD_URL="https://github.com/$GITHUB_REPO/releases/latest/download/${BINARY_NAME}_${OS}_${ARCH}${EXT}"
else
    DOWNLOAD_URL="https://github.com/$GITHUB_REPO/releases/download/${VERSION}/${BINARY_NAME}_${OS}_${ARCH}${EXT}"
fi

echo "Downloading $BINARY_NAME from $DOWNLOAD_URL..."

# Create temporary directory
TMP_DIR=$(mktemp -d)
TMP_FILE="$TMP_DIR/$BINARY_NAME"

# Download binary
if command -v curl > /dev/null 2>&1; then
    curl -sSL "$DOWNLOAD_URL" -o "$TMP_FILE"
elif command -v wget > /dev/null 2>&1; then
    wget -q "$DOWNLOAD_URL" -O "$TMP_FILE"
else
    echo "Error: Neither curl nor wget found. Please install one of them and try again."
    exit 1
fi

# Make binary executable
chmod +x "$TMP_FILE"

# Check if INSTALL_DIR exists in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "Warning: $INSTALL_DIR is not in your PATH. You may need to add it manually."
fi

# Check if we need sudo
if [ -w "$INSTALL_DIR" ]; then
    # We can write to the install dir without sudo
    mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
else
    # We need sudo
    echo "Installing to $INSTALL_DIR (requires sudo)..."
    sudo mv "$TMP_FILE" "$INSTALL_DIR/$BINARY_NAME"
fi

# Clean up
rm -rf "$TMP_DIR"

echo "$BINARY_NAME has been installed to $INSTALL_DIR/$BINARY_NAME"
echo "Run '$BINARY_NAME --help' to get started"
