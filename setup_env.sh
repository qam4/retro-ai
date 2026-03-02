#!/bin/bash
# Source this file to set up the Python environment for development:
#   source setup_env.sh
#
# Uses MinGW Python (matches the MinGW-compiled native module).
# Packages are installed via: pacman -S mingw-w64-x86_64-python-<name>

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${REPO_ROOT}/python;${REPO_ROOT}/build/dev-mingw${PYTHONPATH:+;$PYTHONPATH}"

# Use MinGW Python — the pyenv MSVC Python can't load MinGW-built .pyd files
alias python='C:/msys64/mingw64/bin/python.exe'
alias pytest='C:/msys64/mingw64/bin/python.exe -m pytest'

echo "Environment ready. Using MinGW Python 3.14."
echo "  python -c \"import retro_ai\"  should work now."
echo "  Install packages with: pacman -S mingw-w64-x86_64-python-<name>"
