# MimicVision AI

Sistema multimodal de vision por computador desarrollado como proyecto integral en tres
entregas: interpreta posturas y gestos en tiempo real (MIMIC), recupera contenido visual
similar (MATCH) y permite consultar eventos dentro de un video con lenguaje natural (ASK).

Estado actual: **v1.0 — proyecto completo (Clases 1, 2 y 3)**.

## Que hace cada modulo

**MIMIC (Clase 1).** Detecta landmarks de cuerpo, rostro y manos con MediaPipe Holistic,
los convierte en 10 caracteristicas geometricas invariantes a escala, y clasifica 10
categorias (5 poses y 5 gestos) en tiempo real sobre webcam o video, con etiqueta
suavizada, confianza y FPS en pantalla.

Las 10 clases: `neutral`, `zen`, `pensando`, `brazos_cruzados`, `brazos_abiertos`,
`saludo`, `pulgar_arriba`, `mano_en_menton`, `manos_juntas`, `senalamiento`.

**MATCH (Clase 2).** Recupera imagenes visualmente similares a una consulta, comparando
tres representaciones: descriptores clasicos (HOG+LBP), la geometria heredada de MIMIC,
y embeddings modernos (SigLIP2). Indice por similitud coseno con Top-5 y evaluacion
Recall@1/Recall@5/MRR.

**ASK (Clase 3).** Integra MIMIC y MATCH sobre un video: construye una linea temporal de
eventos, localiza regiones por lenguaje natural con LocateAnything-3B (requiere GPU), y
responde consultas en texto libre con timestamp y evidencia. Funciona en modo archivo y
en modo en vivo (webcam).

## Instalacion local

Requiere Python 3.11 o superior.

```bash
git clone https://github.com/mickldu/mimicvision-ai-2026.git
cd mimicvision-ai-2026
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

Si la maquina no tiene GPU (el caso mas comun en local), conviene instalar antes la
version de PyTorch solo-CPU -- es mucho mas liviana que la version con CUDA que instala
`requirements.txt` por defecto:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

El modelo de landmarks de MediaPipe (~14 MB), los pesos de SigLIP2 (~800 MB) y el
dataset se descargan solos la primera vez que se usan; no estan versionados en el
repositorio (ver "Datos y modelos"). El grounding de ASK con LocateAnything-3B (3.8B
parametros) requiere GPU con arquitectura Ampere o superior (A100/H100) y Linux -- no
corre en Windows local ni en GPUs mas antiguas; ver la limitacion documentada en
`docs/informe_D3_ask.md`.

## Uso en Google Colab

Abrir el notebook de la clase que corresponda (`D1_mimic_baseline.ipynb`,
`D2_match_retrieval.ipynb` o `D3_ask_video.ipynb`) en Colab. La primera celda detecta el
entorno, clona el repositorio e instala las dependencias automaticamente. Las demos usan
la camara del navegador, asi que funcionan igual que en local.

## Como ejecutar

| Que | Comando |
|---|---|
| Notebook de MIMIC (dataset, entrenamiento, metricas, demo) | abrir `notebooks/D1_mimic_baseline.ipynb` |
| Notebook de MATCH (3 rutas, evaluacion, ablation) | abrir `notebooks/D2_match_retrieval.ipynb` |
| Notebook de ASK (timeline, grounding, motor de consulta) | abrir `notebooks/D3_ask_video.ipynb` |
| Demo web completa (Gradio: MIMIC + MATCH + ASK) | `python -m app.app` |
| Demo de escritorio con FPS reales (OpenCV, solo local) | `python -m mimic.live_demo` |
| Tests | `pytest` |

La demo web y la de escritorio necesitan el modelo de MIMIC ya entrenado
(`models/mimic_clasificador.joblib`), que genera `D1_mimic_baseline.ipynb`.

## Estructura de carpetas

```
mimicvision-ai/
├── mimic/                  # Clase 1: percepcion y clasificacion
│   ├── capture.py          #   lectura de frames desde video o webcam
│   ├── landmarks.py        #   envoltorio de MediaPipe HolisticLandmarker
│   ├── features.py         #   vector de 10 features geometricas + bbox de persona
│   ├── classifier.py       #   entrenamiento y seleccion SVM vs Random Forest
│   ├── temporal.py         #   suavizado de etiquetas por ventana
│   ├── pipeline.py         #   integracion: frame -> PerceptionResult
│   ├── live_demo.py        #   demo de escritorio con FPS
│   └── data_curation/      #   scripts reproducibles del dataset de MIMIC
├── match/                  # Clase 2: recuperacion visual por similitud
│   ├── gallery.py          #   galeria y consultas reutilizando metadata.csv de MIMIC
│   ├── descriptors.py      #   HOG + LBP (ruta clasica)
│   ├── embeddings.py       #   SigLIP2 (ruta moderna)
│   ├── index.py            #   indice de similitud coseno, busqueda Top-K
│   ├── evaluate.py         #   Recall@K y MRR
│   └── reranking.py        #   re-ranking por similitud geometrica
├── ask/                    # Clase 3: video, timeline y grounding
│   ├── ingest.py           #   muestreo de frames (archivo o en vivo)
│   ├── timeline.py         #   consolidacion de eventos preliminares (ruta ligera)
│   ├── grounding.py        #   cliente de LocateAnything-3B (requiere GPU)
│   ├── events.py           #   contrato Event
│   ├── store.py            #   persistencia en event_store.json
│   ├── query_engine.py     #   consulta en lenguaje natural -> Event con evidencia
│   └── data_curation/      #   construccion del video de prueba compuesto
├── app/app.py              # interfaz Gradio (una pestana por modulo)
├── notebooks/               # entregables por clase (D1, D2, D3)
├── tests/                   # suite de pytest
├── data/                     # dataset e inventarios (binarios pesados fuera de git)
├── models/                   # modelos serializados (fuera de git)
└── docs/                      # especificaciones, planes de diseno e informes
```

## Datos y modelos

Ningun binario pesado se sube al repositorio. Todo se reconstruye desde codigo:

- **HaGRID** (licencia CC-BY-SA 4.0): 120 imagenes por clase para `saludo` y
  `pulgar_arriba`, descargadas con `python -m mimic.data_curation.descargar_hagrid`.
- **Openverse** (licencias CC0/CC-BY/CC-BY-SA): fotos curadas para las otras 8 clases de
  MIMIC, descargadas con `python -m mimic.data_curation.descargar_fotos_cc`. Cada foto
  queda registrada con su URL de origen, licencia y atribucion en
  `data/raw/cc_registro.csv`.
- `python -m mimic.data_curation.construir_metadata` genera `data/metadata.csv`, que
  MATCH tambien reutiliza como galeria (train+val) y consultas de evaluacion (test).
- **Pexels** (Pexels License, uso libre personal y comercial): 5 clips de video para el
  video de prueba de ASK, descargados y concatenados con
  `python -m ask.data_curation.construir_video_prueba`. Fuente y licencia de cada clip
  en `data/videos/video_prueba_registro.csv`.
- Los pesos de MediaPipe, SigLIP2 y LocateAnything-3B se descargan automaticamente al
  primer uso.

Limitaciones conocidas y documentadas (ver los informes de cada clase en `docs/`): las
8 clases curadas de MIMIC tienen menos de 100 muestras por clase; el clasificador
resultante muestra sesgo hacia las clases mayoritarias, que se propaga a traves de MATCH
y del timeline de ASK; y el grounding de ASK con LocateAnything-3B no se pudo validar en
un entorno sin GPU Ampere+.

## Informes por entrega

- `docs/informe_D1_mimic.md` — MIMIC: dataset, features, resultados, errores.
- `docs/informe_D2_match.md` — MATCH: comparacion de rutas, ablation study.
- `docs/informe_D3_ask.md` — ASK: timeline, evaluacion, limitaciones de hardware.

Las decisiones de diseno de cada clase estan documentadas en `docs/superpowers/specs/`.
