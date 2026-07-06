# Manual paso a paso — MimicVision AI

Guia completa para instalar, ejecutar y solucionar problemas del proyecto, tanto en una
maquina local como en Google Colab. Escrita para alguien que clona el repositorio por
primera vez y no conoce el codigo.

---

## 1. Requisitos previos

| Requisito | Detalle |
|---|---|
| Python | 3.11 o superior (probado con 3.13) |
| Git | Cualquier version reciente |
| Conexion a internet | La primera ejecucion descarga modelos y dataset (~1 GB en total) |
| Espacio en disco | ~3 GB libres (entorno virtual + modelos + dataset) |
| GPU | **No necesaria** para MIMIC ni MATCH. Solo el grounding de ASK (LocateAnything-3B) exige GPU NVIDIA Ampere o superior con Linux — en la practica, usar Colab para esa parte |
| Camara web | Opcional; solo para las demos en vivo |

## 2. Instalacion local, paso a paso

**Paso 1.** Clonar el repositorio y entrar a la carpeta:

```bash
git clone https://github.com/mickldu/mimicvision-ai-2026.git
cd mimicvision-ai-2026
```

**Paso 2.** Crear y activar un entorno virtual:

```bash
python -m venv .venv
```

- En **Windows** (PowerShell o CMD): `.venv\Scripts\activate`
- En **Windows** (Git Bash): `source .venv/Scripts/activate`
- En **Linux/macOS**: `source .venv/bin/activate`

**Paso 3.** Instalar PyTorch. Si la maquina **no tiene GPU NVIDIA** (el caso mas comun),
instalar primero la version solo-CPU — pesa ~200 MB en vez de varios GB de binarios CUDA
que no se usarian:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Si la maquina si tiene GPU NVIDIA con drivers CUDA, saltar este paso: el `pip install`
del paso 4 instalara la version con CUDA automaticamente.

**Paso 4.** Instalar el resto de dependencias:

```bash
pip install -r requirements.txt
```

**Paso 5.** Verificar la instalacion corriendo los tests:

```bash
pytest
```

Resultado esperado: `45 passed` (con algunos warnings, ninguno critico). Si esto pasa,
el entorno esta listo.

## 3. Uso en Google Colab

1. Abrir [colab.research.google.com](https://colab.research.google.com).
2. Archivo → Abrir notebook → pestana GitHub → pegar
   `https://github.com/mickldu/mimicvision-ai-2026` y elegir el notebook deseado.
3. Ejecutar la primera celda: detecta que esta en Colab, clona el repositorio e instala
   las dependencias sola. No hay que configurar nada a mano.
4. Para el notebook D3 (ASK) con grounding real: Entorno de ejecucion → Cambiar tipo de
   entorno de ejecucion → seleccionar una GPU antes de ejecutar.

Las demos con camara usan la camara del navegador, asi que funcionan igual en Colab que
en local.

## 4. Orden de ejecucion (importante)

Los notebooks tienen dependencias entre si — **ejecutarlos en este orden la primera
vez**:

```
D1_mimic_baseline.ipynb   →   D2_match_retrieval.ipynb   →   D3_ask_video.ipynb
```

| Notebook | Que produce | Quien lo necesita despues |
|---|---|---|
| D1 | `data/metadata.csv`, `data/features_mimic.csv`, `models/mimic_clasificador.joblib` | D2, D3 y las demos |
| D2 | caches de descriptores y embeddings (`data/match_*.npz`) | D3 (reconstruye el indice si no estan) |
| D3 | `data/videos/video_prueba.mp4`, `data/event_store.json` | la pestana ASK de la demo |

Tiempos aproximados de la primera ejecucion (CPU de laptop tipica):

- **D1**: 10-20 min (descarga del dataset + extraccion de landmarks + entrenamiento).
- **D2**: 10-15 min (embeddings SigLIP2 de ~275 imagenes, dos veces por el ablation).
- **D3**: 5-10 min sin GPU (timeline completo; el grounding se salta con un mensaje).

Las ejecuciones siguientes son mucho mas rapidas: todo lo costoso queda cacheado en
disco y los notebooks lo detectan y lo reusan.

## 5. Ejecutar las demos

**Demo web (las tres pestanas — MIMIC, MATCH, ASK):**

```bash
python -m app.app
```

Abrir en el navegador la URL que imprime (normalmente `http://127.0.0.1:7860`).

- **MIMIC**: activar la camara y mantener una pose un par de segundos; muestra la
  etiqueta con su confianza.
- **MATCH**: subir o capturar una foto y pulsar "Buscar"; muestra las 5 imagenes mas
  parecidas de la galeria. La primera busqueda tarda ~1-2 min porque construye el
  indice completo; las siguientes son inmediatas.
- **ASK**: pestana "Video": subir un video (por ejemplo
  `data/videos/video_prueba.mp4`), pulsar "Construir timeline" y luego escribir una
  consulta. Pestana "En vivo": la camara alimenta el timeline en tiempo real.

**Demo de escritorio (solo local, para medir FPS reales):**

```bash
python -m mimic.live_demo
```

Abre una ventana de OpenCV con la camara, la etiqueta y los FPS. Salir con la tecla `q`.

**Requisito de ambas demos:** que exista `models/mimic_clasificador.joblib`. Si no
existe, ejecutar primero el notebook D1 (seccion 4 de este manual).

## 6. Regenerar el dataset desde cero (opcional)

El dataset se descarga solo desde el notebook D1, pero tambien se puede reconstruir por
partes desde la terminal:

```bash
python -m mimic.data_curation.descargar_hagrid       # 240 imagenes de HaGRID
python -m mimic.data_curation.descargar_fotos_cc      # fotos CC de Openverse
python -m mimic.data_curation.construir_metadata      # inventario + split
python -m ask.data_curation.construir_video_prueba    # video de prueba de ASK
```

Los tres scripts de descarga son **reanudables**: si se cortan a mitad de camino, se
vuelven a correr y continuan desde donde iban sin repetir descargas.

## 7. Solucion de problemas

**"No existe models/mimic_clasificador.joblib" al abrir una demo.**
El modelo no viene en el repositorio (los binarios no se versionan). Ejecutar el
notebook `D1_mimic_baseline.ipynb` completo una vez; su seccion 5 genera el archivo.

**La descarga de HaGRID falla con error 429 o 503.**
Es el limite de tasa de la API publica de Hugging Face. El script ya reintenta solo con
esperas crecientes; si aun asi se corta, esperar unos minutos y volver a correrlo — es
reanudable y no repite lo ya descargado.

**Advertencia "You are sending unauthenticated requests to the HF Hub".**
Es inofensiva. Para quitarla (y tener descargas mas rapidas), crear una cuenta gratuita
en huggingface.co, generar un token de lectura en Settings → Access Tokens, y definirlo
antes de correr: `set HF_TOKEN=hf_...` (Windows) o `export HF_TOKEN=hf_...` (Linux).

**"LocateAnything-3B requiere GPU CUDA" en el notebook D3 o la pestana ASK.**
Comportamiento esperado en maquinas sin GPU: el modelo de grounding necesita GPU NVIDIA
Ampere o superior. El resto de ASK (timeline, consultas contra eventos) funciona igual.
Para el grounding real, abrir D3 en Colab con GPU (seccion 3, punto 4).

**El puerto 7860 esta ocupado al lanzar la demo.**
Otra instancia de Gradio quedo corriendo. Cerrarla, o lanzar en otro puerto:
`GRADIO_SERVER_PORT=7861 python -m app.app` (Linux/Git Bash) o
`$env:GRADIO_SERVER_PORT=7861; python -m app.app` (PowerShell).

**Cambie el dataset pero las metricas no cambian.**
Los notebooks cachean lo costoso en disco. Borrar la cache correspondiente y volver a
ejecutar: `data/features_mimic.csv` (features de MIMIC), `data/match_*.npz`
(descriptores y embeddings de MATCH), `data/videos/video_prueba.mp4` (video de ASK).

**La primera busqueda en la pestana MATCH tarda mas de un minuto.**
Normal: en ese momento se construye el indice de embeddings de toda la galeria en CPU.
Las busquedas siguientes dentro de la misma sesion son inmediatas.

**Error de mediapipe al procesar imagenes de tamanos distintos.**
Ya esta resuelto dentro del proyecto (todo frame pasa por un lienzo fijo de 640x640
antes del detector). Si aparece, es senal de que se esta llamando a mediapipe
directamente sin pasar por `mimic.landmarks.DetectorHolistic` — usar siempre esa clase.

**`pytest` no se encuentra despues de instalar.**
El entorno virtual no esta activado. Repetir el paso 2 de la seccion 2 (el prompt de la
terminal debe mostrar `(.venv)` al inicio).

## 8. Documentacion complementaria

- `README.md` — vision general del proyecto y estructura de carpetas.
- `docs/informe_D1_mimic.md`, `docs/informe_D2_match.md`, `docs/informe_D3_ask.md` —
  metodo, resultados reales y limitaciones de cada entrega.
- `docs/superpowers/specs/` — decisiones de diseno documentadas de cada modulo.
- Los tres docx en `docs/` — el enunciado original del proyecto con sus rubricas.
