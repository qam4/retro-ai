#!/bin/bash
# Source this file to set up the Python environment for development:
#   source setup_env.sh
#
# Auto-detects the platform and build directory.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the native module build directory
NATIVE_SO=""
for candidate in \
    "$REPO_ROOT/build/ci-linux" \
    "$REPO_ROOT/build/dev" \
    "$REPO_ROOT/build/release" \
    "$REPO_ROOT/build/dev-mingw" \
    "$REPO_ROOT/build/ci-macos"; do
    if ls "$candidate"/retro_ai_native*.so 2>/dev/null 1>&2 || \
       ls "$candidate"/retro_ai_native*.pyd 2>/dev/null 1>&2; then
        NATIVE_SO="$candidate"
        break
    fi
done

if [ -z "$NATIVE_SO" ]; then
    echo "WARNING: Could not find retro_ai_native module in any build directory."
    echo "  Run: cmake --preset ci-linux && cmake --build build/ci-linux -j4"
    echo "  (or the appropriate preset for your platform)"
else
    echo "Found native module in: $NATIVE_SO"
fi

# Platform-aware path separator
SEP=":"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) SEP=";" ;;
esac

export PYTHONPATH="${REPO_ROOT}/python${SEP}${NATIVE_SO}${PYTHONPATH:+${SEP}$PYTHONPATH}"

# Set default ROM directory if not already set
if [ -z "$RETRO_AI_ROM_DIR" ]; then
    export RETRO_AI_ROM_DIR="$REPO_ROOT/roms"
    echo "RETRO_AI_ROM_DIR set to $RETRO_AI_ROM_DIR (override with your ROM path)"
fi

echo "Environment ready."
echo "  python -c \"import retro_ai_native\" should work now."
echo "  retro-ai train game_profiles/course_automobile_training.yaml"
