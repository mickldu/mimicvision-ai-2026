# ASK (Clase 3) — diseño detallado

Fecha: 2026-07-05
Estado: aprobado por el usuario (docente), pendiente de revisión final del documento escrito.
Depende de: `2026-07-05-mimicvision-overview-design.md`, `2026-07-05-mimic-design.md` (Clase 1,
ya implementada) y `2026-07-05-match-design.md` (Clase 2, ya implementada).

## 1. Alcance de este sub-proyecto

Tercera y última entrega del proyecto integral MimicVision AI (40 pts). Integra MIMIC y
MATCH en una aplicación multimodal que indexa un video, construye una línea temporal de
eventos, localiza regiones por lenguaje natural con LocateAnything-3B, y responde
consultas en texto libre con timestamp y evidencia visual. Incluye además un modo en
vivo (webcam), agregado al alcance original del docx por pedido explícito del docente.
El alcance completo del docx está en
`docs/Proyecto_Integral_MimicVision_Clase_3_ASK_NVIDIA.docx`; aquí solo se documentan
las decisiones de diseño específicas de esta implementación de referencia.

## 2. Restricciones reales de hardware verificadas durante el diseño

Antes de diseñar el módulo de grounding se probó la integración real con
`nvidia/LocateAnything-3B` (no se documentó una API sin validarla primero):

- **Auto-hospedar el modelo** (vía `transformers`, código verificado desde su model
  card) requiere explícitamente GPU **Ampere o superior (A100/H100) y Linux**. El T4
  gratuito de Colab es arquitectura Turing (anterior a Ampere).
- **Consumir el Space público de NVIDIA** (`nvidia/LocateAnything`) vía `gradio_client`
  se probó en vivo con una imagen real: falla para llamadas anónimas con
  `"El GPU solicitado (240s) supera el maximo permitido"`, sin importar los parámetros.
  Requeriría como mínimo una cuenta y token de Hugging Face, sin garantía de que alcance.

**Decisión del docente:** auto-hospedar en Colab T4 de todas formas, aceptando el
riesgo. Si falla o rinde mal, se documenta como limitación real en el informe — mismo
criterio de honestidad ya aplicado en los hallazgos de MATCH. Consecuencia práctica para
esta implementación de referencia: **el módulo de grounding (`ask/grounding.py`) no se
puede probar en este entorno local sin GPU** (a diferencia de MediaPipe y SigLIP2, que sí
se validaron localmente). Se escribe con la API real documentada en el model card de
`nvidia/LocateAnything-3B`, y toda la lógica que lo rodea (ingesta, timeline, motor de
consulta) se prueba con un cliente de grounding simulado (stub), de forma que solo la
llamada real al modelo quede pendiente de validar en Colab.

## 3. Video de prueba: composición con verdad de terreno conocida

En vez de un solo clip, se concatenan 5 videos con licencia libre de Pexels en un único
video de ~82 segundos (dentro del rango de 1 a 5 minutos que exige el docx), cada uno
representando un evento reconocible:

| Segmento | Contenido | Duración aprox. | Fuente (Pexels, licencia libre) |
|---|---|---|---|
| 1 | Persona hablando con gestos de manos | 47s | video 4106314 |
| 2 | Pulgar arriba | 9s | video 8627026 |
| 3 | Brazos cruzados | 9s | video 7686684 |
| 4 | Saludo con la mano | 13s | video 4586958 |
| 5 | Señalando | 4s | video 6974223 |

Como los límites de cada segmento se conocen exactamente (se construyeron por
concatenación), se tiene la **verdad de terreno del timeline sin necesidad de anotación
manual**: se sabe qué segundo corresponde a qué evento. Esto permite evaluar la
consolidación temporal de MIMIC (sección 6) completamente en local. Cada clip se
documenta con su fuente y licencia en `data/videos/video_prueba_registro.csv`, igual que
el registro de licencias de las fotos de MIMIC.

## 4. Arquitectura de dos velocidades

- **Ruta ligera (continua):** el pipeline de MIMIC (`mimic.pipeline.procesar_frame` +
  `SuavizadorTemporal`) corre sobre frames muestreados a 8-15 FPS, generando una
  secuencia de etiquetas de pose/gesto estables por tiempo.
- **Ruta VLM (bajo demanda):** LocateAnything-3B solo se invoca cuando el usuario
  formula una consulta en texto, y solo sobre los keyframes de los eventos ya detectados
  por la ruta ligera — nunca sobre cada frame del video.

Esto aplica igual en modo batch (un video de archivo) y en **modo en vivo** (webcam): la
única diferencia es que en modo en vivo el event store crece continuamente en vez de
procesarse de una vez, y el motor de consulta se ejecuta contra los eventos acumulados
hasta el momento.

## 5. Almacenamiento de eventos

`event_store.json` (no SQLite): un solo video o sesión en vivo genera decenas de
eventos, un archivo JSON es más simple y suficiente — SQLite no se justifica a esta
escala, siguiendo el mismo criterio de simplicidad que llevó a usar similitud coseno en
vez de FAISS en MATCH.

## 6. Evaluación: qué se puede validar en local y qué requiere Colab

| Evaluación | Dónde se valida | Motivo |
|---|---|---|
| Consolidación temporal (tasa de acierto de eventos, error de inicio/fin) | **Local** | Se compara contra la verdad de terreno conocida del video compuesto (sección 3) |
| Recall@5 de búsqueda visual heredada de MATCH | **Local** | Reutiliza el índice ya construido y validado en MATCH |
| Grounding textual (IoU sobre frames anotados, tasa de localización, latencia) | **Colab** | Requiere ejecutar LocateAnything-3B con GPU real |
| Prueba de estrés (oclusiones, distractores) | **Colab** | Depende del grounding real |

Esta división se documenta explícitamente en el notebook y en el informe final — no se
presenta como debilidad oculta sino como el límite honesto de lo que se puede verificar
sin acceso a GPU.

## 7. Estructura de módulos

```
ask/
  data_curation/
    construir_video_prueba.py   # descarga los 5 clips CC y los concatena con limites conocidos
  ingest.py            # lee video (archivo o webcam) -> frames muestreados con timestamp
  timeline.py            # aplica mimic.pipeline + SuavizadorTemporal -> eventos preliminares
  grounding.py              # cliente LocateAnything-3B (API real, solo valida en Colab)
  events.py                   # fusiona evento preliminar + resultado de grounding -> Event
  store.py                       # persistencia event_store.json
  query_engine.py                   # consulta en lenguaje natural -> Event con evidencia
```

`ask/query_engine.py` recibe el cliente de grounding como parámetro (no lo instancia
internamente), para poder probarlo con un stub en los tests y con el cliente real en
Colab — mismo patrón de inyección de dependencias que ya usa `mimic/pipeline.py` con el
detector y el modelo.

## 8. Contrato `Event` (ya definido en el diseño transversal)

```python
Event = {
    "event_id": str,
    "type": str,
    "start_time": str,
    "end_time": str,
    "duration_s": float,
    "query": str,
    "frame_path": str,
    "source": str,  # "MIMIC" | "MIMIC+LocateAnything"
}
```

## 9. Prompts de grounding (mínimo 5, del docx, adaptados al video de prueba)

`"person with arms crossed"` (coincide literalmente con el segmento 3), `"person giving
a thumbs up"` (segmento 2), `"person waving hello"` (segmento 4), `"person pointing"`
(segmento 5), `"all people in the scene"` (detección genérica, aplicable a cualquier
segmento).

## 10. Notebook y demo

`D3_ask_video.ipynb`: detección de entorno (igual patrón que D1/D2, con verificación
extra de GPU disponible antes de intentar cargar LocateAnything-3B), construcción del
video de prueba, timeline de MIMIC con evaluación contra la verdad de terreno, carga de
LocateAnything-3B (con manejo de error si no hay GPU compatible), grounding sobre los 5
prompts mínimos, consolidación en `event_store.json`, motor de consulta interactivo, y
demo Gradio (pestaña ASK: elegir video o modo en vivo, escribir consulta, ver
respuesta con timestamp y frame de evidencia).
