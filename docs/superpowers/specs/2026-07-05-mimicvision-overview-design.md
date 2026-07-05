# MimicVision AI 2026 — diseño transversal (arquitectura compartida)

Fecha: 2026-07-05
Estado: aprobado por el usuario (docente), pendiente de revisión final del documento escrito.

## 1. Propósito

Este documento define la arquitectura compartida por las tres entregas del proyecto de curso **MimicVision AI 2026** (módulo de computación por computador, maestría en IA). El usuario es el docente y necesita una **solución de referencia completa y funcional**, no una plantilla vacía para estudiantes.

El proyecto real (para los maestrantes) exige tres entregables, uno por semana, sobre un mismo repositorio que evoluciona sin romper lo anterior:

| Semana | Clase | Capacidad | Peso |
|---|---|---|---|
| 1 | MIMIC | Pose y gestos en tiempo real | 25 pts |
| 2 | MATCH | Recuperación visual por similitud | 35 pts |
| 3 | ASK | Búsqueda multimodal en video con grounding de lenguaje natural | 40 pts |

El alcance detallado de cada clase (objetivos, actividades, entregables, rúbrica) está en `docs/Proyecto_Integral_MimicVision_Clase_{1,2,3}_*.docx` y no se repite aquí — este documento cubre solo lo que las tres clases comparten.

Dado el tamaño del proyecto, el trabajo se divide en tres sub-proyectos secuenciales (MIMIC → MATCH → ASK), cada uno con su propio spec de diseño y plan de implementación. Este documento es la base común que los tres sub-proyectos asumen.

## 2. Estructura del repositorio

```
mimicvision-ai/
├── README.md
├── requirements.txt
├── environment.yml
├── notebooks/
│   ├── D1_mimic_baseline.ipynb
│   ├── D2_match_retrieval.ipynb
│   └── D3_ask_video.ipynb
├── mimic/
├── match/
├── ask/
├── app/
│   └── app.py
├── data/
├── models/
└── docs/
```

Un solo repositorio, un módulo por carpeta (`mimic/`, `match/`, `ask/`) en lugar de tags de git por versión. Cada módulo expone funciones/clases que tanto su notebook como `app/app.py` importan — la lógica vive en un solo lugar.

## 3. Estrategia de entorno (local + Colab)

- Un único `requirements.txt` instalable con `pip`, válido en local y en Colab.
- Cada notebook detecta el entorno al inicio (`try: import google.colab` vs no) y ajusta instalación de dependencias y rutas en consecuencia.
- Los pesos grandes (embeddings preentrenados, LocateAnything-3B, índices FAISS) nunca se versionan en git. Se descargan con un script documentado (`huggingface_hub` / `torch.hub`), igual en ambos entornos.
- LocateAnything-3B (Clase 3) corre directamente sobre la GPU gratuita de Colab (T4); en local se documenta como opcional/CPU-lento según hardware disponible.

## 4. Interfaz de usuario

- **Gradio** como interfaz principal para las tres demos (una pestaña por módulo dentro de `app/app.py`). Corre igual con `demo.launch()` en local y en una celda de Colab, sin necesidad de túneles.
- Captura de cámara vía `gr.Image(sources=["webcam"])`, que usa JavaScript del navegador y funciona igual en ambos entornos.
- Además, un script `mimic/live_demo.py` con OpenCV puro, solo para medir FPS reales en local sin la latencia de red que introduce Gradio.

## 5. Estrategia de datos

El docente no puede grabarse a sí mismo ni producir video propio, así que la solución de referencia usa datasets públicos reales en vez de captura propia:

| Módulo | Necesidad | Fuente propuesta |
|---|---|---|
| MIMIC | 5 poses + 5 gestos, ≥100 muestras/clase, ≥3 condiciones | HaGRID (gestos de mano, CC-BY) + un dataset de poses corporales (a definir en el spec de MIMIC) |
| MATCH | Galería ≥300 imágenes, ≥6 categorías | Stanford 40 Actions + reutilización de imágenes de MIMIC |
| ASK | Video con eventos reconocibles | Dataset público de video (no un clip propio subido), a definir en el spec de ASK |

Todo dataset se documenta con fuente, licencia y script de descarga/preparación reproducible. Ningún binario crudo se sube al repositorio, solo el código que lo genera y el CSV de metadata.

## 6. Contratos de datos entre módulos

MIMIC, MATCH y ASK se comunican mediante estructuras de datos simples (dict o dataclass), no acoplamiento directo de código:

```python
# Lo que MIMIC entrega y MATCH/ASK consumen
PerceptionResult = {
    "frame": np.ndarray,
    "bbox_persona": tuple,
    "landmarks_normalizados": np.ndarray,
    "etiqueta_pose": str,
    "confianza": float,
    "timestamp": float,
}

# Lo que MATCH entrega y ASK consume
MatchResult = {
    "image_id": str,
    "timestamp": float | None,
    "embedding_id": str,
    "etiqueta_pose": str,
    "ruta_archivo": str,
    "score": float,
}

# Unidad final que expone el motor de consulta de ASK
Event = {
    "event_id": str,
    "type": str,
    "start_time": str,
    "end_time": str,
    "duration_s": float,
    "query": str,
    "frame_path": str,
    "source": str,
}
```

## 7. Esqueleto de módulos

```
mimic/
  capture.py        # webcam/video/gr.Image -> frames RGB
  landmarks.py       # frame -> landmarks (MediaPipe)
  features.py         # landmarks -> vector de features geométricas
  classifier.py        # vector -> (etiqueta, confianza)
  temporal.py            # etiquetas por frame -> etiqueta estable
  pipeline.py              # une todo -> PerceptionResult

match/
  gallery.py         # galería + metadata
  descriptors.py       # HOG / LBP
  embeddings.py          # embedding moderno (SigLIP2/DINOv3)
  index.py                 # FAISS: build(), search() -> MatchResult[]
  evaluate.py                 # Recall@K, MRR, latencia, ablation

ask/
  ingest.py           # video o stream en vivo -> keyframes + timestamps
  grounding.py          # frame + prompt -> LocateAnything-3B -> cajas/puntos
  events.py               # detecciones -> Event (consolidación temporal)
  store.py                  # persistencia (event_store.json / SQLite)
  query_engine.py              # consulta en lenguaje natural -> Event + evidencia
```

## 8. Nota de alcance pendiente: ASK en tiempo real

Además del modo batch sobre un video de 1-5 minutos (como pide el doc de la Clase 3), ASK debe soportar también un **modo en vivo** (webcam), consistente con el pipeline de "dos velocidades" que el propio documento de la Clase 3 sugiere: pose/gestos ligeros a 8-15 FPS de forma continua, y LocateAnything invocado bajo demanda solo sobre keyframes o eventos candidatos. Esto implica que `ask/ingest.py` y `ask/query_engine.py` deben soportar tanto una fuente de video fija como un event store que crece en vivo. El detalle se resuelve en el spec de ASK (sub-proyecto 3).

## 9. Secuenciación

1. **Spec MIMIC** (sub-proyecto 1): dataset, features, clasificador, tiempo real, notebook, demo.
2. **Spec MATCH** (sub-proyecto 2): construido sobre MIMIC.
3. **Spec ASK** (sub-proyecto 3): construido sobre MIMIC + MATCH, incluye modo en vivo.

Cada spec se brainstorm-ea, diseña, planifica e implementa antes de pasar al siguiente.

## 10. Métricas de evaluación (verificadas contra los tres docx)

Cada módulo tiene sus propias métricas obligatorias, ya fijadas por la rúbrica del docente. Se listan aquí para que los tres specs las hereden sin reinterpretarlas:

| Módulo | Métrica | Qué mide | Fuente (docx original) |
|---|---|---|---|
| MIMIC | Precision, recall, F1 por clase | Qué tan bien el clasificador acierta cada pose/gesto individual, no solo el promedio | docx Clase 1, §8 (A8) y rúbrica "Evaluación" |
| MIMIC | Matriz de confusión | Entre qué clases se confunde el modelo — clave para explicar los 5 errores reales que pide el criterio de aceptación | docx Clase 1, §11 y §12 |
| MIMIC | FPS y latencia | Si el pipeline realmente corre en tiempo real, no solo en teoría | docx Clase 1, §8 (A8) |
| MATCH | Recall@1, Recall@5 | De cada consulta, si la imagen correcta aparece en el primer resultado o en el top 5 | docx Clase 2, §5, §9 y rúbrica |
| MATCH | MRR (Mean Reciprocal Rank) | Qué tan arriba en el ranking aparece el resultado correcto, no solo si aparece | docx Clase 2, §5, §9 y rúbrica |
| MATCH | Latencia de indexación y consulta | Costo de construir el índice y de responder cada búsqueda | docx Clase 2, §8 (A3, A8) |
| MATCH | Ablation study | Cuánto aporta cada componente (crop, landmarks, textura, re-ranking) al quitar uno a la vez | docx Clase 2, §8 (A9) y rúbrica |
| ASK | IoU y tasa de éxito IoU≥0.5 | Si la caja que devuelve LocateAnything realmente coincide con la región correcta, no solo si "acertó a ojo" | docx Clase 3, §11 |
| ASK | Precision y recall por caja | Cuando hay varias personas/objetos, si el modelo detecta todos sin inventar de más | docx Clase 3, §11 |
| ASK | Tasa de localización correcta con distractores | Robustez cuando hay elementos que podrían confundir al grounding | docx Clase 3, §11 |
| ASK | Latencia (media, mediana, p95) | Tiempo de respuesta real de una consulta end-to-end, incluyendo casos lentos (p95) | docx Clase 3, §11 |
| ASK | Error de inicio/fin y tasa de evento correcto | Si la consolidación temporal arma bien los eventos (no solo si detecta el frame correcto) | docx Clase 3, §9 y §11 |

Verificación: estas métricas coinciden exactamente con las de los tres documentos de la maestría — no se agrega ni se quita ninguna. En el código, cada función de evaluación llevará un comentario en español explicando qué decisión de diseño motiva medir esa métrica en particular (por ejemplo, por qué F1 por clase y no solo accuracy global, dado el desbalance esperado entre poses fáciles y gestos ambiguos).

## 11. Convención de código

Todo el código del proyecto se comenta en español, sin íconos/emojis, con comentarios naturales que expliquen el porqué (no traducciones línea por línea de lo que ya dice el código).
