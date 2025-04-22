# PowerShell script to install molecules binary

# Configuration
$BinaryName = "molecules"
$GithubRepo = "um-univie/molecules"
$InstallDir = "$env:LOCALAPPDATA\Programs\$BinaryName"

# Parse arguments
$Version = if ($args[0]) { $args[0] } else { "latest" }

# Determine architecture
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "386" }

Write-Host "Detected Architecture: $Arch"

# Construct download URL
if ($Version -eq "latest") {
    $DownloadUrl = "https://github.com/$GithubRepo/releases/latest/download/${BinaryName}_windows_${Arch}.exe"
} else {
    $DownloadUrl = "https://github.com/$GithubRepo/releases/download/${Version}/${BinaryName}_windows_${Arch}.exe"
}

Write-Host "Downloading $BinaryName from $DownloadUrl..."

# Create install directory if it doesn't exist
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Download binary
$BinaryPath = Join-Path $InstallDir "$BinaryName.exe"
Invoke-WebRequest -Uri $DownloadUrl -OutFile $BinaryPath -UseBasicParsing

Write-Host "$BinaryName has been downloaded to $BinaryPath"

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$UserPath;$InstallDir", "User")
    Write-Host "Added $InstallDir to your PATH environment variable"
    
    # Update current session PATH
    $env:PATH = "$env:PATH;$InstallDir"
} else {
    Write-Host "$InstallDir is already in your PATH"
}

Write-Host "Installation complete! Run '$BinaryName --help' to get started"
Write-Host "Note: You may need to restart your terminal for the PATH changes to take effect" 