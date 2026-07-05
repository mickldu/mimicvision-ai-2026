# MimicVision AI

Sistema multimodal de vision por computador desarrollado como proyecto integral en tres
entregas: interpreta posturas y gestos en tiempo real (MIMIC), recupera contenido visual
similar (MATCH) y permite consultar eventos dentro de un video con lenguaje natural (ASK).

Estado actual: **v0.1 — Clase 1 (MIMIC)**.

## Que hace la version actual

- Detecta landmarks de cuerpo, rostro y manos con MediaPipe Holistic.
- Convierte los landmarks en 10 caracteristicas geometricas invariantes a escala.
- Clasifica 10 categorias (5 poses y 5 gestos) con un modelo supervisado clasico.
- Funciona en tiempo real sobre webcam o archivo de video, con etiqueta suavizada
  temporalmente, confianza y FPS en pantalla.

Las 10 clases: `neutral`, `zen`, `pensando`, `brazos_cruzados`, `brazos_abiertos`,
`saludo`, `pulgar_arriba`, `mano_en_menton`, `manos_juntas`, `senalamiento`.

## Instalacion local

Requiere Python 3.11 o superior.

```bash
git clone <URL_DEL_REPOSITORIO>
cd mimicvision-ai
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
repositorio (ver "Datos y modelos").

## Uso en Google Colab

Abrir `notebooks/D1_mimic_baseline.ipynb` en Colab. La primera celda detecta el entorno,
clona el repositorio e instala las dependencias automaticamente. La demo usa la camara
del navegador, asi que funciona igual que en local.

## Como ejecutar

| Que | Comando |
|---|---|
| Notebook completo (dataset, entrenamiento, metricas, demo) | abrir `notebooks/D1_mimic_baseline.ipynb` |
| Demo web (Gradio, camara del navegador) | `python -m app.app` |
| Demo de escritorio con FPS reales (OpenCV, solo local) | `python -m mimic.live_demo` |
| Tests | `pytest` |

Tanto la demo web como la de escritorio necesitan el modelo entrenado
(`models/mimic_clasificador.joblib`), que genera el notebook en su seccion 5.

## Estructura de carpetas

```
mimicvision-ai/
├── mimic/                  # modulo de la Clase 1: percepcion y clasificacion
│   ├── capture.py          #   lectura de frames desde video o webcam
│   ├── landmarks.py        #   envoltorio de MediaPipe HolisticLandmarker
│   ├── features.py         #   vector de 10 features geometricas
│   ├── classifier.py       #   entrenamiento y seleccion SVM vs Random Forest
│   ├── temporal.py         #   suavizado de etiquetas por ventana
│   ├── pipeline.py         #   integracion: frame -> PerceptionResult
│   ├── live_demo.py        #   demo de escritorio con FPS
│   └── data_curation/      #   scripts reproducibles del dataset
├── app/app.py              # interfaz Gradio (una pestana por modulo)
├── notebooks/              # entregables por clase (D1, D2, D3)
├── tests/                  # suite de pytest
├── data/                   # dataset e inventarios (imagenes fuera de git)
├── models/                 # modelos serializados (fuera de git)
└── docs/                   # especificaciones y planes de diseno
```

## Datos y modelos

Ningun binario pesado se sube al repositorio. Todo se reconstruye desde codigo:

- **HaGRID** (licencia CC-BY-SA 4.0): 120 imagenes por clase para `saludo` y
  `pulgar_arriba`, descargadas con `python -m mimic.data_curation.descargar_hagrid`
  desde el subconjunto publico `cj-mills/hagrid-classification-512p-no-gesture-150k`.
- **Openverse** (licencias CC0/CC-BY/CC-BY-SA): fotos curadas para las otras 8 clases,
  descargadas con `python -m mimic.data_curation.descargar_fotos_cc`. Cada foto queda
  registrada con su URL de origen, licencia y atribucion en `data/raw/cc_registro.csv`.
- `python -m mimic.data_curation.construir_metadata` genera `data/metadata.csv` con el
  inventario completo y el split train/val/test estratificado y reproducible.
- El modelo de landmarks de MediaPipe se descarga automaticamente al primer uso.

Limitacion conocida y documentada: las 8 clases curadas de Openverse tienen menos de 100
muestras por clase (se decidio usar solo fotos reales, sin aumento de datos). El analisis
de su efecto esta en el informe de la entrega.

## Continuidad del proyecto

- **Clase 2 (MATCH)** reutilizara `mimic.pipeline.procesar_frame`, que entrega por frame:
  imagen, landmarks normalizados, etiqueta de pose, confianza y timestamp.
- **Clase 3 (ASK)** integrara ambos modulos con grounding por lenguaje natural sobre video.

Las decisiones de diseno estan documentadas en `docs/superpowers/specs/`.
