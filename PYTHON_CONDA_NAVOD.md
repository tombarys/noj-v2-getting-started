# Návod: Conda a Python prostředí pro Clojure+Python projekty

## 🎯 Co je conda a proč ji používat?

**Conda** je správce balíčků a prostředí pro Python (a další jazyky). Umožňuje:
- Vytvářet izolovaná prostředí s různými verzemi Pythonu
- Snadno instalovat balíčky včetně složitých závislostí (numpy, pandas, scikit-learn)
- Spravovat konflikty mezi balíčky
- Přepínat mezi prostředími podle projektů

## 🔍 Aktuální stav tvého systému

```bash
# Conda je nainstalovaná: conda 23.3.1
# Máš tato prostředí:
# - base (výchozí miniconda prostředí) - AKTIVNÍ PRO LIBPYTHON-CLJ
# - myenv (vlastní prostředí)
# - pipenv prostředí (aktuálně aktivní v terminálu)
```

## 📋 Krok za krokem návod

### 1. Vytvoření nového conda prostředí pro tvůj projekt

```bash
# Vytvoř nové prostředí s názvem "noj-ml" a Python 3.10
conda create -n noj-ml python=3.10

# Aktivuj nové prostředí
conda activate noj-ml
```

### 2. Instalace potřebných balíčků

```bash
# Základní data science balíčky
conda install pandas numpy scipy

# Scikit-learn pro machine learning
conda install scikit-learn

# Jupyter pro notebooky (volitelné)
conda install jupyter

# Matplotlib pro grafy (volitelné)
conda install matplotlib seaborn
```

### 3. Ověření instalace

```bash
# Zkontroluj, že je vše nainstalované
python -c "import sklearn; print('sklearn version:', sklearn.__version__)"
python -c "import pandas; print('pandas version:', pandas.__version__)"
python -c "import numpy; print('numpy version:', numpy.__version__)"
```

### 4. Nastavení pro libpython-clj

V tvém Clojure kódu potřebuješ nastavit správnou cestu k Python interpreteru:

```clojure
(py/initialize! {:python-executable "/Users/tomas/miniconda3/envs/noj-ml/bin/python"})
```

## 🔧 Užitečné conda příkazy

### Správa prostředí
```bash
# Seznam všech prostředí
conda env list

# Aktivace prostředí
conda activate název-prostředí

# Deaktivace (návrat do base)
conda deactivate

# Smazání prostředí
conda env remove -n název-prostředí
```

### Správa balíčků
```bash
# Instalace balíčku
conda install název-balíčku

# Instalace konkrétní verze
conda install pandas=2.2.3

# Seznam nainstalovaných balíčků
conda list

# Aktualizace balíčku
conda update název-balíčku

# Odebrání balíčku
conda remove název-balíčku
```

### Export/Import prostředí
```bash
# Export prostředí do souboru
conda env export > environment.yml

# Vytvoření prostředí ze souboru
conda env create -f environment.yml
```

## 🚀 Doporučený workflow pro tvůj projekt

### Varianta A: Používej conda prostředí (doporučeno)

1. **Vytvořit conda prostředí:**
```bash
conda create -n noj-ml python=3.10
conda activate noj-ml
conda install pandas numpy scipy scikit-learn
```

2. **V Clojure kódu:**
```clojure
(py/initialize! {:python-executable "/Users/tomas/miniconda3/envs/noj-ml/bin/python"})
```

### Varianta B: Použít stávající base prostředí

1. **Aktivovat base a doinstalovat:**
```bash
conda activate base
conda install pandas numpy scipy scikit-learn
```

2. **V Clojure kódu:**
```clojure
(py/initialize!) ; použije /Users/tomas/miniconda3/bin/python
```

## ⚠️ Časté problémy a řešení

### Problem 1: "No module named 'sklearn'"
**Řešení:** Ujisti se, že jsi v správném prostředí a scikit-learn je nainstalován
```bash
conda activate tvoje-prostředí
conda install scikit-learn
```

### Problem 2: Nekompatibilní verze numpy/pandas
**Řešení:** Nech conda vyřešit závislosti
```bash
conda install pandas scikit-learn  # conda automaticky vybere kompatibilní verze
```

### Problem 3: libpython-clj používá špatný Python
**Řešení:** Explicitně nastav cestu
```clojure
(py/initialize! {:python-executable "/cesta/k/python"})
```

### Problem 4: Aktivace prostředí nefunguje
**Řešení:** Reinitiuj conda v shellu
```bash
conda init zsh  # pro zsh shell
# restart terminal
```

## 🔍 Jak zjistit, které prostředí používá libpython-clj

```clojure
; V Clojure REPL
(py/run-simple-string "import sys; print(sys.executable)")
```

## 📁 Struktura conda instalace

```
/Users/tomas/miniconda3/           # Base conda instalace
├── bin/python                     # Python z base prostředí
├── envs/                         # Složka s virtuálními prostředími
│   ├── noj-ml/                   # Tvoje nové prostředí
│   │   └── bin/python            # Python z tohoto prostředí
│   └── myenv/                    # Existující prostředí
└── pkgs/                         # Cache stažených balíčků
```

## 💡 Pro-tipy

1. **Používej environment.yml** pro reprodukovatelnost:
```yaml
name: noj-ml
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - scipy
```

2. **Nastav alias** pro rychlé přepnutí:
```bash
# V ~/.zshrc
alias activate-noj="conda activate noj-ml"
```

3. **Používej conda místo pip** pro data science balíčky (lepší správa závislostí)

4. **Pravidelně aktualizuj** prostředí:
```bash
conda update --all
```

## 🎯 Doporučení pro tvůj konkrétní případ

Protože máš funkční setup s base prostředím, doporučuji:

1. **Krátkodobě:** Pokračuj s base prostředím, které už funguje
2. **Dlouhodobě:** Vytvoř dedikované prostředí pro data science projekty

**Quick fix pro aktuální projekt:**
```bash
# Ujisti se, že base má vše potřebné
conda activate base
conda install pandas=2.2.3 scikit-learn -y
```

**V Clojure kódu neměň nic** - libpython-clj už používá správné prostředí!
