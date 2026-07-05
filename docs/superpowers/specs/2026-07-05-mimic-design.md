# MIMIC (Clase 1) — diseño detallado

Fecha: 2026-07-05
Estado: aprobado por el usuario (docente), pendiente de revisión final del documento escrito.
Depende de: `2026-07-05-mimicvision-overview-design.md` (arquitectura compartida, contratos de datos, estructura de repo).

## 1. Alcance de este sub-proyecto

Primera entrega del proyecto integral MimicVision AI (25 pts). Clasifica 10 categorías en tiempo real (5 poses corporales + 5 gestos de mano) a partir de landmarks de MediaPipe, y expone el resultado como `PerceptionResult` (definido en el diseño transversal) para que MATCH y ASK lo reutilicen. El alcance completo (objetivos, actividades obligatorias, criterios de aceptación, rúbrica) está en `docs/Proyecto_Integral_MimicVision_Clase_1_MIMIC.docx`; aquí solo se documentan las decisiones de diseño específicas de esta implementación de referencia.

## 2. Dataset (10 clases)

El docente no puede grabar video propio, así que el dataset combina dos fuentes reales:

| Clase | Tipo | Fuente | Volumen objetivo |
|---|---|---|---|
| saludo | gesto | HaGRID (clase `palm`) | ≥100 |
| pulgar arriba | gesto | HaGRID (clase `like`) | ≥100 |
| mano en el mentón | gesto | Fotos de bancos CC curadas a mano | 20-40 |
| manos juntas | gesto | Fotos de bancos CC curadas a mano | 20-40 |
| señalamiento | gesto | Fotos de bancos CC curadas a mano | 20-40 |
| neutral | pose | Fotos de bancos CC curadas a mano | 20-40 |
| zen | pose | Fotos de bancos CC curadas a mano | 20-40 |
| pensando | pose | Fotos de bancos CC curadas a mano | 20-40 |
| brazos cruzados | pose | Fotos de bancos CC curadas a mano | 20-40 |
| brazos abiertos | pose | Fotos de bancos CC curadas a mano | 20-40 |

Decisión explícita: **sin aumento de datos**. Las 8 clases curadas a mano quedarán por debajo del mínimo recomendado de 100 muestras/clase que sugiere el docx. Esta es una limitación real y se documenta abiertamente en el informe final (E6, máximo 3 páginas), no se oculta ni se compensa artificialmente.

`data/metadata.csv` registra por imagen: `sample_id, clase, fuente, licencia, url_origen, participante_o_condicion, split`.

## 3. Percepción: MediaPipe Holistic

Se usa **MediaPipe Holistic** (pose + manos + rostro en una sola pasada) en vez de detectores separados, porque varias clases dependen de la relación entre partes del cuerpo, no de una sola parte aislada. Nota técnica verificada durante la implementación: la API vigente es `mediapipe.tasks.python.vision.HolisticLandmarker` (paquete `mediapipe` 0.10.35), no el módulo antiguo `mp.solutions.holistic` que ya no existe en esta versión. Requiere descargar el modelo `holistic_landmarker.task` (~14 MB) desde el repositorio de modelos de Google — se documenta como descarga, no se versiona en git. Devuelve hasta 33 puntos de pose, 478 de rostro y 21 por mano detectada.

- "mano en el mentón" → distancia mano-rostro
- "manos juntas" → distancia mano-mano

- "mano en el mentón" → distancia mano-rostro
- "manos juntas" → distancia mano-mano
- "señalamiento" → ángulo brazo-antebrazo + orientación de mano
- "brazos cruzados" / "brazos abiertos" → posición de manos respecto al torso

## 4. Features geométricas (mínimo 10, normalizadas)

Ángulo de codo (izquierdo y derecho), ángulo de hombro (izquierdo y derecho), ángulo de rodilla (izquierdo y derecho), inclinación de cabeza, distancia mano-mano, distancia mano-mentón, distancia mano-hombro contrario, apertura de brazos relativa al ancho de hombros. Todo normalizado por la distancia hombro-cadera de cada persona, para que la escala de la imagen no afecte la clasificación.

## 5. Clasificador y evaluación

- Se entrenan y comparan **SVM (kernel RBF)** y **Random Forest** — ambos sin necesidad de GPU, según el stack sugerido en el docx.
- Split estratificado 70/15/15 (train/val/test) por clase.
- Dado que las 8 clases curadas tienen pocas muestras (20-40), el hold-out de test por sí solo no es confiable — se complementa con **cross-validation estratificada (k-fold)** para reportar una métrica más estable.
- Selección del modelo final por **F1 macro en validación**, no accuracy global, porque las clases están desbalanceadas entre las 2 clases HaGRID (cientos de muestras) y las 8 curadas (decenas de muestras).
- Evaluación final sobre el set de test: matriz de confusión, precision/recall/F1 por clase, y análisis de 5 errores reales.

## 6. Tiempo real y FPS

`mimic/pipeline.py` aplica el modelo sobre frames en vivo (webcam local, `gr.Image(sources=["webcam"])` en Gradio, o archivo de video), con suavizado temporal por ventana (mayoría móvil sobre las últimas N predicciones) para evitar parpadeo de etiqueta. `mimic/live_demo.py` (script OpenCV puro) mide FPS reales en local, sin la latencia de red de Gradio.

## 7. Estructura del notebook `D1_mimic_baseline.ipynb`

1. Detección de entorno (Colab/local) e instalación de dependencias.
2. Descarga/preparación del dataset (script de curación de fotos CC + subset de HaGRID).
3. Extracción de landmarks con MediaPipe Holistic.
4. Ingeniería de features geométricas.
5. Split estratificado y entrenamiento (SVM vs Random Forest).
6. Evaluación: matriz de confusión, F1 por clase, cross-validation.
7. Selección y serialización del modelo final (`.joblib`).
8. Integración en tiempo real vía `mimic/pipeline.py` (overlay de landmarks, etiqueta, confianza, FPS).
9. Demo Gradio (`app/app.py`, pestaña MIMIC).
10. Análisis de 5 errores reales con propuestas concretas de mejora.

## 8. Salida hacia MATCH y ASK

`mimic/pipeline.py` produce `PerceptionResult` (definido en el diseño transversal) por frame: `frame`, `bbox_persona`, `landmarks_normalizados`, `etiqueta_pose`, `confianza`, `timestamp`. Este es el contrato que MATCH (Clase 2) y ASK (Clase 3) consumirán sin necesidad de tocar el código de MIMIC.
