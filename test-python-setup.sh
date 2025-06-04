#!/bin/bash

# 🧪 Test skript pro ověření Python/conda setupu

echo "🔍 Testování Python prostředí pro Clojure projekty"
echo "=================================================="

echo
echo "📍 1. Informace o conda:"
conda --version
echo

echo "📍 2. Seznam conda prostředí:"
conda env list
echo

echo "📍 3. Aktuální Python v terminálu:"
which python
python --version
echo

echo "📍 4. Base conda Python:"
echo "Cesta: /Users/tomas/miniconda3/bin/python"
/Users/tomas/miniconda3/bin/python --version
echo

echo "📍 5. Test balíčků v base prostředí:"
/Users/tomas/miniconda3/bin/python -c "
try:
    import sklearn, pandas, numpy, scipy
    print('✅ sklearn:', sklearn.__version__)
    print('✅ pandas:', pandas.__version__)
    print('✅ numpy:', numpy.__version__)
    print('✅ scipy:', scipy.__version__)
    print()
    print('🎉 Všechny balíčky jsou dostupné!')
except ImportError as e:
    print('❌ Chyba importu:', e)
"
echo

echo "📍 6. Doporučení pro libpython-clj:"
echo "V Clojure kódu použij:"
echo "(py/initialize! {:python-executable \"/Users/tomas/miniconda3/bin/python\"})"
echo
echo "Nebo jednoduše:"
echo "(py/initialize!)  ; pokud base prostředí je výchozí"
echo

echo "📍 7. Jak přepnout prostředí v budoucnu:"
echo "conda activate název-prostředí"
echo "conda deactivate"
echo

echo "✨ Test dokončen!"
