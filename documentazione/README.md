# Plottery - Documentazione

## Panoramica

Plottery è una libreria Python per estrarre dati numerici da immagini di grafici scientifici (spettri, curve, istogrammi, scatter plot, barre impilate) utilizzando Claude Vision API.

## Installazione

```bash
pip install plottery
```

Oppure da sorgente:

```bash
git clone https://github.com/your-repo/plottery
cd plottery
pip install -e .
```

## Configurazione

Plottery richiede una API key di Anthropic:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Oppure crea un file `.env` nella root del progetto:

```
ANTHROPIC_API_KEY=your-api-key
```

## Utilizzo Rapido

### Con PDF

```python
from plottery import Paper

# Carica PDF ed estrai tutti i grafici
paper = Paper("paper.pdf")
paper.extract_all()

# Mostra risultati
for chart in paper.charts:
    print(f"Page {chart.page}: {chart.type}")
    print(f"  Series: {chart.num_series}")
    print(f"  Points: {chart.num_points}")

# Esporta
paper.to_excel("output.xlsx")  # Un foglio per grafico
paper.to_json("output.json")   # Tutto in JSON
paper.to_csv("output/")        # Un CSV per grafico
```

### Con singola immagine

```python
from plottery import Chart

# Carica immagine
chart = Chart.from_image("spectrum.png")

# Estrai dati (con contesto opzionale)
chart.extract(context="Motor frequency spectrum")

# Accedi ai dati
print(f"Type: {chart.type}")
print(f"Series: {chart.num_series}")
print(f"Points: {chart.num_points}")

# Esporta
chart.to_csv("data.csv")
df = chart.to_dataframe()  # pandas DataFrame
```

### Configurazione globale

```python
from plottery import config

# Modifica densità di campionamento
config.sample_density = "high"  # "low", "medium", "high"

# Disabilita rilevamento picchi
config.detect_peaks = False
```

## Classi Principali

### Paper

Rappresenta un documento PDF con i suoi grafici.

```python
class Paper:
    path: Path              # Percorso del PDF
    text: str               # Testo estratto dal PDF
    charts: List[Chart]     # Lista dei grafici trovati

    def extract_all(
        self,
        context=None,           # Contesto opzionale
        generate_context=True,  # Genera contesto dal paper
        max_workers=1,          # 1=sequenziale, >1=parallelo
        on_progress=None,       # Callback (completed, total, chart)
    ) -> Paper

    def to_csv(self, output_dir: str) -> List[Path]
    def to_excel(self, path: str) -> None
    def to_json(self, path: str) -> None
    def save_images(self, output_dir: str) -> List[Path]

    # Properties
    num_charts: int         # Numero di grafici
    extracted_charts: List[Chart]  # Grafici estratti con successo
    total_points: int       # Punti totali
    total_series: int       # Serie totali
```

#### Estrazione parallela

```python
# Estrazione sequenziale (default)
paper.extract_all()

# Estrazione parallela con 4 worker
paper.extract_all(max_workers=4)

# Con callback di progresso
def on_progress(completed, total, chart):
    print(f"[{completed}/{total}] Page {chart.page}: {chart.type}")

paper.extract_all(max_workers=4, on_progress=on_progress)
```

### Chart

Rappresenta un singolo grafico con capacità di estrazione.

```python
class Chart:
    image: np.ndarray       # Immagine RGB
    page: int               # Numero pagina (se da PDF)
    type: str               # Tipo rilevato (line, bar, scatter, etc.)
    context: str            # Contesto usato per estrazione
    series: List[Series]    # Serie estratte
    peaks: List[Peak]       # Picchi rilevati
    metadata: Dict          # Metadata aggiuntivi
    x_range: Tuple[float, float]  # Range asse X
    y_range: Tuple[float, float]  # Range asse Y
    x_unit: str             # Unità asse X
    y_unit: str             # Unità asse Y

    @classmethod
    def from_image(cls, path: str) -> Chart

    def extract(self, context=None, x_unit=None, y_unit=None) -> Chart
    def to_csv(self, path: str) -> None
    def to_dataframe(self) -> pd.DataFrame
    def save_image(self, path: str) -> None

    # Properties
    is_extracted: bool      # Se i dati sono stati estratti
    is_categorical: bool    # Se asse X è categorico
    categories: List[str]   # Categorie (se categorico)
    num_points: int         # Punti totali
    num_series: int         # Numero serie
```

### Series

Rappresenta una serie di dati estratta.

```python
@dataclass
class Series:
    name: str               # Nome della serie
    color: Tuple[int, int, int]  # Colore RGB
    points: List[Tuple[float, float]]  # Punti (x, y)

    def to_dataframe(self) -> pd.DataFrame
    def to_numpy(self) -> np.ndarray

    # Properties
    x: np.ndarray           # Coordinate X
    y: np.ndarray           # Coordinate Y
```

### Peak

Rappresenta un picco rilevato.

```python
@dataclass
class Peak:
    x: float                # Coordinata X
    y: float                # Coordinata Y (ampiezza)
    series: str             # Nome serie
    prominence: float       # Prominenza
    harmonic_of: float      # Frequenza fondamentale (se armonico)
    harmonic_order: int     # Ordine armonico

    # Properties
    is_harmonic: bool       # Se è un armonico
```

### Config

Configurazione centralizzata.

```python
@dataclass
class Config:
    api_key: str            # API key Anthropic
    model: str              # Modello Claude (default: claude-opus-4-5-20251101)
    sample_density: str     # "low", "medium", "high"
    detect_peaks: bool      # Abilita rilevamento picchi

    # Properties
    is_configured: bool     # Se API key è impostata
```

## Tipi di Grafici Supportati

| Tipo | Descrizione | Asse X |
|------|-------------|--------|
| **line** | Grafici a linea, curve continue | Numerico |
| **spectrum** | Spettri di frequenza | Numerico |
| **bar** | Grafici a barre | Numerico o Categorico |
| **stacked_bar** | Grafici a barre impilate | Categorico |
| **histogram** | Istogrammi | Numerico |
| **scatter** | Scatter plot (punti) | Numerico |
| **pie** | Grafici a torta | Categorico |
| **area** | Grafici ad area | Numerico |

## Gestione Assi Categorici

Per grafici con asse X categorico (es. "Anal.", "IEC"):

```python
chart = Chart.from_image("stacked_bar.png")
chart.extract()

if chart.is_categorical:
    categories = chart.categories  # ["Anal.", "IEC"]
    mapping = chart.category_mapping  # {0: "Anal.", 1: "IEC"}

    # I punti usano indici numerici
    # series.points = [(0.0, 173.8), (1.0, 187.4)]
```

## Demo

Vedi la cartella `demo/` per esempi di utilizzo:

```bash
# Attiva virtual environment
source venv/bin/activate

# Demo semplice con nuova API
python demo/demo_simple.py

# Demo con generazione automatica contesto
python demo/demo_auto_context.py
```

## Struttura File

```
plottery/
├── __init__.py         # API pubblica: Paper, Chart, config
├── paper.py            # Classe Paper
├── chart.py            # Classe Chart
├── models.py           # Series, Peak
├── config.py           # Configurazione
└── utils/
    ├── pdf.py          # Elaborazione PDF
    └── peaks.py        # Rilevamento picchi
```

## Dipendenze

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "pillow>=10.0",
    "anthropic>=0.18",
]

[project.optional-dependencies]
pdf = ["pymupdf>=1.23"]
peaks = ["scipy>=1.11"]
excel = ["openpyxl>=3.1"]
all = ["plottery[pdf,peaks,excel]"]
```

## Test

```bash
# Esegui tutti i test
pytest tests/

# Con copertura
pytest tests/ --cov=plottery

# Test specifico
pytest tests/test_chart.py
```

## Note

- Plottery usa Claude Vision API per leggere i grafici
- I dati vengono estratti direttamente dall'immagine senza preprocessing
- Il contesto (dal paper o fornito manualmente) migliora l'accuratezza
- La densità di campionamento controlla il numero di punti estratti
