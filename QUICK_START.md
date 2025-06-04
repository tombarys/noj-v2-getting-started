# 🚀 Rychlý start - Python/Conda pro tvůj projekt

## ✅ Aktuální stav (FUNGUJE!)
Tvoje conda base prostředí má vše potřebné:
- Python 3.10.10
- scikit-learn 1.6.1  
- pandas 2.2.3
- numpy 2.2.5
- scipy 1.15.3

## 🎯 Jak to používat

### V Clojure kódu:
```clojure
(py/initialize!)  ; Používá automaticky /Users/tomas/miniconda3/bin/python
```

### Ověření v terminálu:
```bash
./test-python-setup.sh
```

## 🔧 Hlavní conda příkazy

```bash
# Seznam prostředí
conda env list

# Aktivace prostředí  
conda activate název-prostředí

# Instalace balíčku
conda install název-balíčku

# Vytvoření nového prostředí
conda create -n nové-prostředí python=3.10

# Vytvoření prostředí ze souboru
conda env create -f environment.yml
```

## 🆘 Když něco nefunguje

1. **Restart terminálu** a zkus znovu
2. **Zkontroluj prostředí:** `conda env list`
3. **Reinstaluj balíček:** `conda install scikit-learn -y`
4. **Spusť test:** `./test-python-setup.sh`

## 📚 Další informace
Detailní návod: `PYTHON_CONDA_NAVOD.md`
