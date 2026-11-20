@echo off
chcp 65001 >nul
echo ============================================
echo   AI-Vision-Guardian 一键安装脚本 (Win11)
echo ============================================
echo.

REM 统一使用项目内 Python 3.11 虚拟环境，避免误用系统其他 Python
if exist ".venv\Scripts\python.exe" goto use_venv
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未检测到 Python 3.11，请先安装 Python 3.11
    pause
    exit /b 1
)
py -3.11 -m venv .venv

:use_venv
set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" --version

REM 检测是否有 NVIDIA GPU
echo [1/3] 检测 NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo       已检测到 NVIDIA GPU ✓
    echo.
    echo [2/3] 安装 CUDA 12.4 版 PyTorch（GPU 加速）...
    "%PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
) else (
    echo       未检测到 NVIDIA GPU
    echo [2/3] 安装 CPU 版 PyTorch（无 GPU 加速）...
    "%PYTHON%" -m pip install torch torchvision torchaudio
)

echo.
echo [3/3] 安装其他依赖...
"%PYTHON%" -m pip install -r requirements.txt

echo.
echo ============================================
echo   安装完成！
echo ============================================
echo.

REM 验证 PyTorch CUDA 状态
"%PYTHON%" -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'GPU: 无 (CPU 模式)')"

echo.
pause
