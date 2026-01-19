# Plottery - User Guide

## Indice

1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Esempi Pratici](#esempi-pratici)
6. [Configurazione](#configurazione)
7. [Demo](#demo)
8. [Tipi di Grafici](#tipi-di-grafici)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Introduzione

**Plottery** è una libreria Python che estrae dati numerici da immagini di grafici scientifici utilizzando Claude Vision API.

### Caratteristiche principali

- **Estrazione da PDF**: Carica un paper scientifico e estrai automaticamente tutti i grafici
- **Estrazione da immagini**: Lavora con singole immagini PNG/JPG
- **Contesto automatico**: Genera contesto dal testo del paper per migliorare l'accuratezza
- **Supporto multi-formato**: Esporta in CSV, Excel, JSON, DataFrame
- **Estrazione parallela**: Velocizza l'elaborazione con worker multipli
- **Rilevamento picchi**: Identifica automaticamente picchi e armoniche

### Tipi di grafici supportati

| Tipo | Descrizione |
|------|-------------|
| `line` | Grafici a linea, curve continue |
| `spectrum` | Spettri di frequenza (FFT, ecc.) |
| `bar` | Grafici a barre |
| `stacked_bar` | Grafici a barre impilate |
| `histogram` | Istogrammi |
| `scatter` | Scatter plot |
| `pie` | Grafici a torta |
| `area` | Grafici ad area |

---

## Installazione

### Da PyPI (quando disponibile)

```bash
pip install plottery
```

### Da sorgente (sviluppo)

```bash
git clone https://github.com/your-repo/plottery
cd plottery
pip install -e ".[all]"
```

### Dipendenze opzionali

```bash
# Solo PDF
pip install plottery[pdf]

# Solo picchi
pip install plottery[peaks]

# Solo Excel
pip install plottery[excel]

# Tutto
pip install plottery[all]
```

### Configurazione API Key

Plottery richiede una API key di Anthropic. Opzioni:

**Opzione 1: Variabile d'ambiente**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Opzione 2: File .env**
```
ANTHROPIC_API_KEY=sk-ant-...
```

**Opzione 3: Da codice**
```python
from plottery import config
config.api_key = "sk-ant-..."
```

---

## Quick Start

### Esempio minimo: PDF

```python
from plottery import Paper

# Carica PDF ed estrai tutti i grafici
paper = Paper("paper.pdf")
paper.extract_all()

# Esporta
paper.to_csv("output/")
```

### Esempio minimo: Immagine singola

```python
from plottery import Chart

# Carica immagine ed estrai dati
chart = Chart.from_image("spectrum.png")
chart.extract()

# Usa come DataFrame
df = chart.to_dataframe()
print(df.head())
```

---

## API Reference

### Classe `Paper`

Rappresenta un documento PDF con i suoi grafici.

#### Costruttore

```python
Paper(
    path: str,              # Percorso del PDF
    dpi: int = 200,         # Risoluzione estrazione immagini
    pages: List[int] = None # Pagine da processare (None = tutte)
)
```

#### Metodi

| Metodo | Descrizione | Ritorna |
|--------|-------------|---------|
| `extract_all(context=None, generate_context=True, max_workers=1, on_progress=None)` | Estrae dati da tutti i grafici | `Paper` |
| `to_csv(output_dir)` | Esporta ogni grafico in CSV separato | `List[Path]` |
| `to_excel(path)` | Esporta in Excel (un foglio per grafico) | `None` |
| `to_json(path)` | Esporta tutto in JSON | `None` |
| `save_images(output_dir)` | Salva le immagini dei grafici | `List[Path]` |
| `get_chart(index)` | Ottiene un grafico per indice | `Chart` |
| `get_charts_on_page(page)` | Ottiene grafici di una pagina | `List[Chart]` |

#### Proprietà

| Proprietà | Tipo | Descrizione |
|-----------|------|-------------|
| `path` | `Path` | Percorso del PDF |
| `text` | `str` | Testo estratto dal PDF |
| `charts` | `List[Chart]` | Lista dei grafici trovati |
| `num_charts` | `int` | Numero di grafici |
| `extracted_charts` | `List[Chart]` | Grafici estratti con successo |
| `total_points` | `int` | Punti totali estratti |
| `total_series` | `int` | Serie totali estratte |

---

### Classe `Chart`

Rappresenta un singolo grafico con capacità di estrazione.

#### Factory Methods

```python
# Da file immagine
chart = Chart.from_image("path/to/image.png")

# Da array numpy
chart = Chart.from_array(numpy_array, page=0)
```

#### Metodi

| Metodo | Descrizione | Ritorna |
|--------|-------------|---------|
| `extract(context=None, x_unit=None, y_unit=None, chart_type_hint=None)` | Estrae dati dal grafico | `Chart` |
| `to_csv(path, sep=",")` | Esporta in CSV | `None` |
| `to_dataframe()` | Converte in DataFrame | `pd.DataFrame` |
| `save_image(path)` | Salva l'immagine | `None` |
| `get_series(name)` | Ottiene una serie per nome | `Series` |
| `get_peaks_for_series(name)` | Ottiene picchi di una serie | `List[Peak]` |

#### Proprietà

| Proprietà | Tipo | Descrizione |
|-----------|------|-------------|
| `image` | `np.ndarray` | Immagine RGB |
| `page` | `int` | Numero pagina (se da PDF) |
| `type` | `str` | Tipo rilevato (line, bar, etc.) |
| `context` | `str` | Contesto usato per estrazione |
| `series` | `List[Series]` | Serie estratte |
| `peaks` | `List[Peak]` | Picchi rilevati |
| `metadata` | `Dict` | Metadata aggiuntivi |
| `x_range` | `Tuple[float, float]` | Range asse X |
| `y_range` | `Tuple[float, float]` | Range asse Y |
| `x_unit` | `str` | Unità asse X |
| `y_unit` | `str` | Unità asse Y |
| `is_extracted` | `bool` | Se i dati sono stati estratti |
| `is_categorical` | `bool` | Se asse X è categorico |
| `categories` | `List[str]` | Nomi categorie (se categorico) |
| `category_mapping` | `Dict[int, str]` | Mapping indice -> categoria |
| `num_points` | `int` | Punti totali |
| `num_series` | `int` | Numero serie |

---

### Classe `Series`

Rappresenta una serie di dati estratta.

```python
@dataclass
class Series:
    name: str                          # Nome della serie
    color: Tuple[int, int, int]        # Colore RGB
    points: List[Tuple[float, float]]  # Punti (x, y)
```

#### Metodi

| Metodo | Descrizione | Ritorna |
|--------|-------------|---------|
| `to_dataframe()` | Converte in DataFrame | `pd.DataFrame` |
| `to_numpy()` | Converte in array numpy | `np.ndarray` |

#### Proprietà

| Proprietà | Tipo | Descrizione |
|-----------|------|-------------|
| `x` | `np.ndarray` | Coordinate X |
| `y` | `np.ndarray` | Coordinate Y |

---

### Classe `Peak`

Rappresenta un picco rilevato.

```python
@dataclass
class Peak:
    x: float                    # Coordinata X
    y: float                    # Coordinata Y (ampiezza)
    series: str                 # Nome serie
    prominence: float = 0.0     # Prominenza
    harmonic_of: float = None   # Frequenza fondamentale (se armonico)
    harmonic_order: int = None  # Ordine armonico
```

#### Proprietà

| Proprietà | Tipo | Descrizione |
|-----------|------|-------------|
| `is_harmonic` | `bool` | Se è un armonico |

---

### Classe `Config`

Configurazione centralizzata (singleton).

```python
from plottery import config

# Proprietà modificabili
config.api_key = "sk-ant-..."           # API key Anthropic
config.model = "claude-opus-4-5-20251101"  # Modello Claude
config.sample_density = "medium"        # "low", "medium", "high"
config.detect_peaks = True              # Abilita rilevamento picchi

# Proprietà di sola lettura
config.is_configured  # True se API key è impostata
```

---

## Esempi Pratici

### 1. Estrazione base da PDF

```python
from plottery import Paper

# Carica e processa
paper = Paper("research_paper.pdf")
paper.extract_all()

# Mostra risultati
print(f"Grafici trovati: {paper.num_charts}")
print(f"Punti estratti: {paper.total_points}")

for i, chart in enumerate(paper.charts):
    print(f"\nGrafico {i+1} (pagina {chart.page + 1}):")
    print(f"  Tipo: {chart.type}")
    print(f"  Serie: {chart.num_series}")
    print(f"  Punti: {chart.num_points}")
    if chart.x_unit:
        print(f"  Unità X: {chart.x_unit}")
    if chart.y_unit:
        print(f"  Unità Y: {chart.y_unit}")
```

### 2. Estrazione parallela con progress

```python
from plottery import Paper

def on_progress(completed, total, chart):
    status = "OK" if chart.is_extracted else "FAIL"
    print(f"[{completed}/{total}] Page {chart.page+1}: {chart.type} [{status}]")

paper = Paper("big_document.pdf")
paper.extract_all(max_workers=4, on_progress=on_progress)
```

### 3. Estrazione con contesto personalizzato

```python
from plottery import Chart

chart = Chart.from_image("motor_spectrum.png")
chart.extract(
    context="FFT spectrum of motor current showing harmonics at 50Hz fundamental frequency",
    x_unit="Hz",
    y_unit="dB",
    chart_type_hint="spectrum"
)

# Accedi ai picchi
for peak in chart.peaks:
    print(f"Peak at {peak.x} Hz: {peak.y} dB")
    if peak.is_harmonic:
        print(f"  -> {peak.harmonic_order}° harmonic of {peak.harmonic_of} Hz")
```

### 4. Elaborazione di grafici categorici

```python
from plottery import Chart

chart = Chart.from_image("stacked_bar_chart.png")
chart.extract()

if chart.is_categorical:
    print(f"Categorie: {chart.categories}")
    # es. ["Analytical", "FEM", "Experimental"]

    for series in chart.series:
        print(f"\nSerie: {series.name}")
        for x, y in series.points:
            category = chart.category_mapping.get(int(x), f"Cat {int(x)}")
            print(f"  {category}: {y}")
```

### 5. Analisi di spettri con armoniche

```python
from plottery import Chart
from plottery.utils.peaks import analyze_spectrum

chart = Chart.from_image("fft_spectrum.png")
chart.extract(context="Motor vibration FFT spectrum")

# Analisi dettagliata dei picchi
for series in chart.series:
    analysis = analyze_spectrum(series, fundamental_hint=50.0)

    print(f"\nSerie: {series.name}")
    print(f"  Picchi trovati: {analysis['peak_count']}")
    print(f"  Armoniche trovate: {analysis['harmonic_count']}")
    print(f"  Fondamentale: {analysis['fundamental']} Hz")
    print(f"  Noise floor: {analysis['noise_floor']:.1f} dB")
```

### 6. Esportazione in vari formati

```python
from plottery import Paper

paper = Paper("paper.pdf")
paper.extract_all()

# CSV: un file per grafico
csv_files = paper.to_csv("output/csv/")
print(f"Creati {len(csv_files)} file CSV")

# Excel: un foglio per grafico + metadati
paper.to_excel("output/all_charts.xlsx")

# JSON: tutto in un file strutturato
paper.to_json("output/all_charts.json")

# Immagini: salva le immagini dei grafici
image_files = paper.save_images("output/images/")
```

### 7. Accesso ai dati come DataFrame

```python
from plottery import Chart

chart = Chart.from_image("line_chart.png")
chart.extract()

# DataFrame con tutte le serie
df = chart.to_dataframe()
# Colonne: x, y, series

# DataFrame di una singola serie
series = chart.get_series("Motor Current")
df_single = series.to_dataframe()
# Colonne: x, y

# Come numpy array
arr = series.to_numpy()  # shape (n, 2)
x_vals = series.x        # array 1D
y_vals = series.y        # array 1D
```

### 8. Elaborazione solo di alcune pagine

```python
from plottery import Paper

# Solo pagine 3, 5, 7 (0-indexed: 2, 4, 6)
paper = Paper("paper.pdf", pages=[2, 4, 6])
paper.extract_all()

# Ottieni grafici di una specifica pagina
page_5_charts = paper.get_charts_on_page(4)
```

### 9. Gestione errori

```python
from plottery import Paper

paper = Paper("paper.pdf")
paper.extract_all()

for chart in paper.charts:
    if chart.is_extracted:
        print(f"Page {chart.page+1}: OK - {chart.num_points} points")
    else:
        error = chart.metadata.get("error", "Unknown error")
        print(f"Page {chart.page+1}: FAILED - {error}")
```

---

## Configurazione

### Densità di campionamento

Controlla quanti punti vengono estratti per serie:

| Valore | Punti per serie | Uso consigliato |
|--------|-----------------|-----------------|
| `"low"` | 10-20 | Quick preview, grafici semplici |
| `"medium"` | 30-50 | Default, buon bilanciamento |
| `"high"` | 50-100+ | Massima precisione, grafici dettagliati |

```python
from plottery import config

config.sample_density = "high"  # Prima di estrarre
```

### Rilevamento picchi

Abilita/disabilita il rilevamento automatico dei picchi:

```python
config.detect_peaks = True   # Abilita (default)
config.detect_peaks = False  # Disabilita
```

### Modello Claude

Cambia il modello usato (default: `claude-opus-4-5-20251101`):

```python
config.model = "claude-sonnet-4-20250514"  # Più veloce, meno accurato
```

---

## Demo

Il progetto include tre demo nella cartella `demo/`:

### `demo_simple.py` - Esempio minimo

L'utilizzo più semplice possibile di Plottery:

```bash
python demo/demo_simple.py
```

```python
from plottery import Paper

paper = Paper("paper.pdf")
paper.extract_all()
paper.to_csv("output/")
```

### `demo_auto_context.py` - Contesto automatico

Mostra la generazione automatica del contesto dal testo del paper:

```bash
python demo/demo_auto_context.py
```

Questo demo:
1. Carica un PDF con `Paper()`
2. Estrae il testo del paper
3. Genera automaticamente contesto per ogni grafico
4. Mostra il contesto generato

### `demo_full.py` - Demo completa

Demo avanzata con tutte le funzionalità:

```bash
# Modalità base
python demo/demo_full.py

# Con debug (salva immagini, contesto, metadata)
python demo/demo_full.py --debug

# Estrazione parallela
python demo/demo_full.py --parallel 4

# Entrambi
python demo/demo_full.py --debug --parallel 4
```

Questo demo include:
- Estrazione da PDF
- Estrazione da singola immagine
- Dimostrazione configurazione
- Modalità debug con salvataggio metadati
- Estrazione parallela

---

## Tipi di Grafici

### Line Chart / Spectrum

Grafici a linea continua, curve, spettri di frequenza.

```python
chart.extract(chart_type_hint="line")
# oppure
chart.extract(chart_type_hint="spectrum")
```

**Caratteristiche estratte**:
- Punti (x, y) lungo la curva
- Range assi
- Picchi (se abilitato)
- Armoniche (per spettri)

### Bar Chart

Grafici a barre verticali o orizzontali.

```python
chart.extract(chart_type_hint="bar")
```

**Nota**: L'asse X può essere numerico o categorico. Plottery rileva automaticamente.

### Stacked Bar Chart

Grafici a barre impilate con componenti multiple.

```python
chart.extract(chart_type_hint="stacked_bar")
```

**Caratteristiche**:
- Ogni componente dello stack è una serie separata
- Asse X tipicamente categorico
- Valori sono le altezze dei singoli componenti

### Scatter Plot

Grafici a dispersione (punti).

```python
chart.extract(chart_type_hint="scatter")
```

### Histogram

Istogrammi di distribuzioni.

```python
chart.extract(chart_type_hint="histogram")
```

### Pie Chart

Grafici a torta.

```python
chart.extract(chart_type_hint="pie")
```

**Nota**: Per i grafici a torta, x = indice categoria, y = percentuale/valore.

---

## Best Practices

### 1. Usa sempre il contesto

Il contesto migliora significativamente l'accuratezza:

```python
# Buono
chart.extract(context="Motor current spectrum showing 50Hz fundamental and harmonics")

# Meglio (con unità)
chart.extract(
    context="Motor current spectrum",
    x_unit="Hz",
    y_unit="dB"
)
```

### 2. Usa `generate_context=True` con i PDF

Quando estrai da PDF, lascia che Plottery generi il contesto dal testo:

```python
paper.extract_all(generate_context=True)  # Default
```

### 3. Usa estrazione parallela per molti grafici

```python
# Per PDF con molti grafici
paper.extract_all(max_workers=4)
```

### 4. Verifica sempre i risultati

```python
for chart in paper.charts:
    if not chart.is_extracted:
        print(f"Warning: Chart on page {chart.page+1} failed")
    elif chart.num_points < 10:
        print(f"Warning: Chart on page {chart.page+1} has few points")
```

### 5. Usa la densità appropriata

- `"low"`: Preview veloce, stima generale
- `"medium"`: Default, bilanciato
- `"high"`: Analisi dettagliata, ricostruzione accurata

### 6. Salva sempre le immagini durante il debug

```python
paper.save_images("debug/images/")
```

---

## Troubleshooting

### Errore: "API key not configured"

**Causa**: ANTHROPIC_API_KEY non impostata.

**Soluzione**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

O nel codice:
```python
from plottery import config
config.api_key = "sk-ant-..."
```

### Errore: "LLM extraction failed"

**Cause possibili**:
1. Immagine di qualità troppo bassa
2. Grafico troppo complesso
3. Timeout API

**Soluzioni**:
1. Usa DPI più alto: `Paper("file.pdf", dpi=300)`
2. Fornisci contesto dettagliato
3. Riprova l'estrazione

### Pochi punti estratti

**Causa**: Densità troppo bassa o grafico poco leggibile.

**Soluzioni**:
```python
config.sample_density = "high"
```

### Grafici non trovati nel PDF

**Cause possibili**:
1. Grafici vettoriali (non immagini)
2. DPI troppo basso

**Soluzioni**:
1. Converti PDF in immagini esternamente
2. Aumenta DPI: `Paper("file.pdf", dpi=300)`

### Errore con grafici categorici

**Problema**: I valori x sono numeri invece di categorie.

**Soluzione**: Usa il mapping:
```python
if chart.is_categorical:
    for x, y in series.points:
        category = chart.category_mapping[int(x)]
        print(f"{category}: {y}")
```

### Import error: "anthropic not found"

**Causa**: Pacchetto anthropic non installato.

**Soluzione**:
```bash
pip install anthropic
# oppure
pip install plottery[all]
```

---

## Struttura del Progetto

```
plottery/
├── plottery/
│   ├── __init__.py      # API pubblica
│   ├── paper.py         # Classe Paper
│   ├── chart.py         # Classe Chart
│   ├── models.py        # Series, Peak
│   ├── config.py        # Configurazione
│   └── utils/
│       ├── pdf.py       # Elaborazione PDF
│       └── peaks.py     # Rilevamento picchi
├── demo/
│   ├── demo_simple.py      # Esempio minimo
│   ├── demo_auto_context.py # Contesto automatico
│   └── demo_full.py        # Demo completa
├── tests/
│   └── ...
├── documentazione/
│   ├── README.md        # Documentazione base
│   └── USERGUIDE.md     # Questa guida
└── pyproject.toml
```

---

## Changelog

### v0.2.0
- Nuova API semplificata con classi `Paper` e `Chart`
- Rimossa dipendenza da Computer Vision (solo LLM)
- Aggiunto supporto estrazione parallela
- Aggiunta generazione automatica contesto

### v0.1.0
- Release iniziale
- Estrazione basata su CV + LLM
