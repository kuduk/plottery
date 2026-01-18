# Plottery Demo

Questa cartella contiene esempi di utilizzo della libreria Plottery.

## Demo disponibili

### demo_llm_extraction.py (Raccomandato)

Demo dell'approccio LLM-based per l'estrazione dati. Passa le immagini direttamente a Claude che legge i valori dal grafico.

**Esecuzione:**
```bash
cd /home/kuduk/plottery
source venv/bin/activate
python demo/demo_llm_extraction.py
```

**Cosa fa:**
1. Estrae le immagini dei grafici dal PDF
2. Filtra le immagini che non sono grafici
3. Passa ogni grafico a Claude con contesto dal paper
4. Claude legge direttamente i valori dagli assi
5. Restituisce dati strutturati (serie, picchi, calibrazione)
6. Salva in CSV

**Vantaggi:**
- Più semplice: niente computer vision complessa
- Più accurato: Claude capisce il contesto
- Funziona con grafici di bassa qualità
- Rileva automaticamente scale e unità

### demo_auto_context.py

Demo del workflow completo con generazione automatica del contesto:
1. Estrae il testo dal paper PDF
2. Per ogni grafico, Claude legge il paper e genera contesto specifico
3. Usa il contesto generato per estrarre i dati con maggiore accuratezza

**Esecuzione:**
```bash
cd /home/kuduk/plottery
source venv/bin/activate
python demo/demo_auto_context.py
```

**Output aggiuntivo:**
- `chart_X_context.txt` - Contesto generato per ogni grafico

### demo_frosini2010.py

Demo dell'approccio tradizionale con computer vision + VLM per classificazione.

**Esecuzione:**
```bash
cd /home/kuduk/plottery
source venv/bin/activate
python demo/demo_frosini2010.py
```

**Cosa fa:**
1. Estrae automaticamente i grafici dal PDF
2. Classifica ogni grafico (line chart, bar chart, etc.)
3. Rileva automaticamente la scala degli assi usando VLM (Claude API)
4. Estrae i punti dati con segmentazione colore
5. Rileva i picchi negli spettri
6. Salva tutto in CSV nella cartella `output/`

**Output:**
- `chart_X_pageY.png` - Immagini dei grafici estratti
- `chart_X_data.csv` - Dati estratti in formato CSV (approccio CV)
- `chart_X_llm_data.csv` - Dati estratti in formato CSV (approccio LLM)

## Struttura

```
demo/
├── README.md
├── demo_frosini2010.py     # Script demo principale
├── data/
│   ├── images/
│   │   └── demo.png        # Immagine di test
│   └── papers/
│       └── frosini2010.pdf # Paper di esempio
└── output/                 # Output generato
    ├── chart_1_data.csv
    ├── chart_2_data.csv
    └── ...
```

## Risultati esempio

### Paper Frosini 2010 (spettri frequenza)
- **Grafici estratti**: 7
- **Punti dati totali**: ~500
- **Picchi rilevati**: ~70

Scale rilevate automaticamente:
- Spettri di frequenza: X = 0-2000 Hz, Y = -180 a 0 dB
- Curve di efficienza: X = 3-7.5 Nm, Y = 79-87%

### Paper IEMDC (perdite motore)
- **Grafici estratti**: 4
- **Punti dati totali**: 64
- **Tipi**: scatter, line, stacked_bar

Esempio grafico a barre impilate:
```
Ps (stator copper losses): Anal.=173.8W, IEC=187.4W
Pr (rotor copper losses):  Anal.=120.5W, IEC=117.6W
Pmech (mechanical losses): Anal.=21.5W,  IEC=28.5W
PFe (iron losses):         Anal.=228.0W, IEC=206.1W
PLL (additional losses):   Anal.=66.9W,  IEC=30.8W
```

## Utilizzo base

```python
import plottery

# Estrazione semplice
result = plottery.extract("chart.png")

# Con calibrazione manuale
result = plottery.extract("spectrum.png", calibration={
    "x_range": (0, 2000),   # Hz
    "y_range": (-180, 0),   # dB
})

# Accesso ai dati
df = result.to_dataframe()
result.to_csv("output.csv")

# Analisi picchi
for peak in result.peaks:
    print(f"{peak.x:.1f} Hz: {peak.y:.1f} dB")
```
