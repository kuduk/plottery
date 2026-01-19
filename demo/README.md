# Plottery - Demo

Questa cartella contiene esempi pratici di utilizzo di Plottery.

## File Demo

| File | Descrizione | Complessità |
|------|-------------|-------------|
| `demo_simple.py` | Esempio minimo (~30 righe) | Basico |
| `demo_auto_context.py` | Generazione automatica contesto | Intermedio |
| `demo_full.py` | Demo completa con tutte le opzioni | Avanzato |

---

## `demo_simple.py` - Esempio Minimo

L'utilizzo più semplice possibile di Plottery:

```bash
python demo/demo_simple.py
```

**Cosa fa**:
1. Carica un PDF
2. Estrae tutti i grafici
3. Stampa i risultati
4. Esporta in CSV

**Codice chiave**:
```python
from plottery import Paper

paper = Paper("paper.pdf")
paper.extract_all()
paper.to_csv("output/")
```

---

## `demo_auto_context.py` - Contesto Automatico

Dimostra la generazione automatica del contesto dal testo del paper:

```bash
python demo/demo_auto_context.py
```

**Cosa fa**:
1. Carica un PDF con `Paper()`
2. Estrae il testo completo del paper
3. Per ogni grafico, genera automaticamente un contesto
4. Mostra il contesto generato per ogni grafico
5. Esporta i risultati

**Uso consigliato**: Quando vuoi vedere come Plottery interpreta i grafici in base al contenuto del paper.

---

## `demo_full.py` - Demo Completa

Demo avanzata con tutte le funzionalità e opzioni CLI:

```bash
# Aiuto
python demo/demo_full.py --help

# Modalità normale
python demo/demo_full.py

# Con debug (salva immagini, metadati, contesto)
python demo/demo_full.py --debug

# Estrazione parallela con 4 worker
python demo/demo_full.py --parallel 4

# Entrambe le opzioni
python demo/demo_full.py --debug --parallel 4
```

**Opzioni CLI**:
- `--debug` / `-d`: Salva immagini, file JSON con metadati, e file txt con contesto
- `--parallel N` / `-p N`: Usa N worker per estrazione parallela

**Cosa include**:
1. **Demo 1**: Estrazione da PDF
   - Progress callback
   - Estrazione parallela opzionale
   - Riepilogo risultati
2. **Demo 2**: Estrazione da singola immagine
   - Contesto manuale
   - Accesso ai dati serie
3. **Demo 3**: Configurazione
   - Mostra opzioni disponibili

---

## Struttura Output

### Modalità normale

```
demo/output/demo_simple/
├── chart_1_page2.csv
├── chart_2_page3.csv
└── paper_data.json
```

### Modalità debug

```
demo/output/demo_full/
├── chart_1_page2.csv
├── paper_data.json
└── debug/
    ├── chart_1_page2.png           # Immagine grafico
    ├── chart_1_page2_debug.json    # Metadati completi
    └── chart_1_page2_context.txt   # Contesto generato
```

---

## Dati di Test

La cartella `data/` contiene:

```
data/
├── images/
│   └── demo.png                    # Immagine spettro esempio
└── papers/
    ├── frosini2010.pdf             # Paper di test principale
    └── ...                         # Altri paper
```

---

## Come Scegliere la Demo

| Situazione | Demo consigliata |
|------------|------------------|
| Prima volta con Plottery | `demo_simple.py` |
| Capire come funziona il contesto | `demo_auto_context.py` |
| Testare tutte le funzionalità | `demo_full.py` |
| Debug problemi di estrazione | `demo_full.py --debug` |
| Velocizzare elaborazione | `demo_full.py --parallel 4` |

---

## Requisiti

Prima di eseguire le demo:

1. **API Key**: Assicurati che `ANTHROPIC_API_KEY` sia impostata in `.env` o come variabile d'ambiente

2. **Dipendenze**: Installa le dipendenze
   ```bash
   pip install -e ".[all]"
   ```

3. **PDF di test**: Verifica che `data/papers/frosini2010.pdf` esista

---

## Output Atteso

### `demo_simple.py`

```
Found 8 charts, extracted 1247 points
  Page 2: line - 2 series, 156 pts
  Page 3: bar - 3 series, 24 pts
  ...

Data exported to demo/output/simple/
```

### `demo_full.py --debug`

```
============================================================
Demo 1: Extract charts from PDF
============================================================

Loading: frosini2010.pdf
Text extracted: 45623 characters
Charts found: 8

Extracting data from all charts (sequential)...
  [1/8] Chart page 2: spectrum OK
  [2/8] Chart page 3: bar OK
  ...

Extraction time: 45.3s

Chart 1 (page 2):
  Type: spectrum
  Series: 2
  Points: 156
  X range: (0.0, 2000.0)
  [debug] Image: chart_1_page2.png
  [debug] Info: chart_1_page2_debug.json

Exporting...
CSV files: 8
JSON: paper_data.json
Debug files: demo/output/demo_full/debug
```

---

## Utilizzo Base (API)

```python
from plottery import Paper, Chart

# === Da PDF ===
paper = Paper("paper.pdf")
paper.extract_all()

for chart in paper.charts:
    print(f"Page {chart.page+1}: {chart.type}")
    print(f"  Series: {chart.num_series}")
    print(f"  Points: {chart.num_points}")

paper.to_excel("output.xlsx")

# === Da immagine singola ===
chart = Chart.from_image("spectrum.png")
chart.extract(context="Motor frequency spectrum")
df = chart.to_dataframe()

# === Configurazione ===
from plottery import config
config.sample_density = "high"  # più punti
config.detect_peaks = False     # disabilita picchi
```
