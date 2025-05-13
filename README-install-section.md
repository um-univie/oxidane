# Installation

You can install `oxidane` with a single command:

## macOS and Linux

```bash
curl -sSL https://raw.githubusercontent.com/um-univie/oxidane/main/install.sh | bash
```

This will download the latest version. To install a specific version:

```bash
curl -sSL https://raw.githubusercontent.com/um-univie/oxidane/main/install.sh | bash -s -- v1.0.0
```

## Windows

Open PowerShell and run:

```powershell
iwr -useb https://raw.githubusercontent.com/um-univie/oxidane/main/install.ps1 | iex
```

For a specific version:

```powershell
iwr -useb https://raw.githubusercontent.com/um-univie/oxidane/main/install.ps1 | iex -args "v1.0.0"
```

## Manual Installation

If you prefer to manually install:

1. Download the appropriate binary for your platform from the [releases page](https://github.com/um-univie/oxidane/releases)
2. Rename it to `oxidane` (or `oxidane.exe` on Windows)
3. Make it executable (on macOS/Linux): `chmod +x oxidane`
4. Move it to a directory in your PATH:
   - macOS/Linux: `/usr/local/bin` or `~/bin`
   - Windows: Create a directory and add it to your PATH environment variable

## Verifying Installation

After installation, verify it works by running:

```bash
oxidane --version
``` 