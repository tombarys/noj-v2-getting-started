#!/bin/bash

# Skript pro spuštění Clojure REPL s správně nastaveným Python prostředím
# Použití: ./start-clojure-repl.sh

echo "🚀 Starting Clojure REPL with Python noj-ml environment..."

# Nastavení cesty k Pythonu v prostředí noj-ml
export PYTHON_EXECUTABLE=/opt/homebrew/Caskroom/miniconda/base/envs/noj-ml/bin/python

# Ověření, že Python a sklearn jsou dostupné
echo "🔍 Verifying Python environment..."
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "❌ ERROR: Python not found at $PYTHON_EXECUTABLE"
    echo "Please check if conda environment 'noj-ml' exists and is properly configured."
    exit 1
fi

echo "✓ Python executable: $PYTHON_EXECUTABLE"
echo "✓ Python version: $($PYTHON_EXECUTABLE --version)"

# Test sklearn
if ! $PYTHON_EXECUTABLE -c "import sklearn; print('sklearn version:', sklearn.__version__)" 2>/dev/null; then
    echo "❌ ERROR: sklearn not available in Python environment"
    echo "Please install sklearn: conda activate noj-ml && conda install scikit-learn"
    exit 1
fi

echo "✓ sklearn is available"
echo ""
echo "🎯 Environment variables set:"
echo "   PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE"
echo ""
echo "📝 Now you can:"
echo "   1. Start Calva REPL in VSCode"
echo "   2. Load your notebook files"
echo "   3. Use sklearn-clj without ModuleNotFoundError"
echo ""
echo "Starting Clojure REPL..."
echo ""

# Spuštění Clojure REPL
clj
