# =====================================================================
# Janelia MaleCNS 3D Game - Prerequisite Installation Script for Windows
# =====================================================================

Write-Host "[*] Checking Python environment..." -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Error: Python is not installed or added to PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[*] Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "[*] Installing core dependencies (FastAPI, Uvicorn, Pandas, Requests, NeuPrint)..." -ForegroundColor Cyan
pip install fastapi uvicorn pandas numpy requests neuprint-python

Write-Host "[*] Re-installing llama-cpp-python with CUDA support for RTX 3060 Mobile..." -ForegroundColor Cyan
Write-Host "[*] Note: This requires the CUDA Toolkit / PyTorch runtime to be active on your machine." -ForegroundColor Yellow

# Force uninstall any existing CPU-only version
pip uninstall llama-cpp-python -y

# Set CMake compilation flags to activate CUDA GGML backend
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

Write-Host "[*] Installation complete! All prerequisites are ready." -ForegroundColor Green