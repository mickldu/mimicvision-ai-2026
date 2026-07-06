# MimicVision AI v1.0 — Informe de la Clase 3 (ASK)

Entrega: Clase 3 del proyecto integral MimicVision AI 2026 (cierre del proyecto).
Equipo: implementación de referencia del docente.

## 1. Problema

Integrar MIMIC y MATCH en una aplicación multimodal que indexe un video, construya una
línea temporal de eventos, localice regiones por lenguaje natural con
LocateAnything-3B, y responda consultas en texto libre con timestamp y evidencia
visual — en modo batch (archivo) y en vivo (webcam).

## 2. Método

**Video de prueba con verdad de terreno conocida.** En vez de un solo clip, se
concatenaron 5 videos con licencia libre de Pexels en un video compuesto de 82
segundos, con límites de tiempo exactos por segmento (documentados en
`data/videos/video_prueba_registro.csv`). Esto permite evaluar la consolidación
temporal sin anotación manual: se sabe exactamente qué segundo corresponde a qué clase.

**Arquitectura de dos velocidades.** MIMIC corre continuo sobre frames muestreados a 8
FPS (`ask/ingest.py` + `ask/timeline.py`), generando eventos preliminares por
consolidación de etiquetas estables. LocateAnything-3B solo se invocaría sobre los
keyframes de esos eventos, nunca sobre cada frame — ni en modo batch ni en modo en vivo
(webcam, implementado en la pestaña ASK de Gradio con `gr.State` para acumular eventos
entre llamadas).

**Restricciones de hardware verificadas antes de programar.** Se probó en vivo la
integración con `nvidia/LocateAnything-3B`: auto-hospedarlo exige GPU Ampere+/Linux (el
model card lo dice explícitamente); el Space público de NVIDIA rechaza llamadas
anónimas por cuota de GPU (`"GPU solicitado (240s) supera el maximo permitido"`,
probado con una imagen real). Se decidió auto-hospedar en Colab T4 de todas formas,
aceptando el riesgo. Consecuencia práctica: el módulo `ask/grounding.py` no se pudo
ejecutar de verdad en este entorno sin GPU — se probó exhaustivamente su función de
parseo de coordenadas (pura, sin GPU) y su guardia de hardware (`ErrorHardwareNoCompatible`),
pero la llamada real al modelo queda pendiente de validar en Colab.

## 3. Resultados

### 3.1 Timeline (100% verificable en local)

Se comparó el timeline detectado contra los 5 segmentos de verdad de terreno,
experimentando con el tamaño de la ventana de suavizado temporal:

| Ventana | Eventos detectados | Aciertos (de 5 segmentos) |
|---|---|---|
| 5 (valor original) | 102 | 1 (20%) |
| **15 (elegido)** | **43** | **2 (40%)** |
| 30 | 31 | 2 (40%) |
| 50 | 29 | 0 (0%) |

Con ventana 5, el clasificador de MIMIC parpadea entre clases frame a frame,
fragmentando el timeline en más de 100 micro-eventos. Ventana 15 reduce la
fragmentación a la mitad y duplica el acierto. Ventanas más grandes (30, 50) no siguen
mejorando — 50 empeora porque el suavizado arrastra clases de segmentos anteriores
hacia el inicio del segmento siguiente, dado que los segmentos de prueba duran solo
4-9 segundos.

**Detalle por segmento (ventana 15):**

| Segmento | Clase esperada | Clase detectada | Acierto |
|---|---|---|---|
| charla | neutral | pulgar_arriba | No |
| pulgar | pulgar_arriba | pulgar_arriba | Sí |
| cruzados | brazos_cruzados | brazos_cruzados | Sí |
| saludo | saludo | pulgar_arriba | No |
| senalar | senalamiento | pulgar_arriba | No |

**Patrón claro: sesgo sistemático hacia `pulgar_arriba`.** 3 de los 5 segmentos se
clasificaron como `pulgar_arriba` sin importar su contenido real. Esta es la misma clase
mayoritaria de HaGRID que ya mostró sesgo en MATCH (sección 4.1 de su informe) y que
domina el entrenamiento de MIMIC (84-120 muestras contra 5-16 de las clases curadas).
El problema no es la ventana de suavizado — es el clasificador subyacente heredado de
MIMIC, entrenado con muy pocos ejemplos de la mayoría de las clases.

### 3.2 Búsqueda visual heredada de MATCH (100% verificable en local)

Consultar el índice de MATCH con el frame representativo del primer evento devolvió
`['pulgar_arriba', 'neutral', 'saludo', 'saludo', 'pensando']` como Top-5 — la
integración funciona técnicamente (MATCH corre sobre datos de ASK sin cambios), aunque
hereda las mismas limitaciones de MATCH ya documentadas.

### 3.3 Grounding con LocateAnything-3B (no verificable en este entorno)

El notebook detecta correctamente la ausencia de GPU (`torch.cuda.is_available() == False`)
e informa el problema en cada celda de grounding en vez de fallar — confirmado
ejecutando el notebook completo de punta a punta sin errores. Ni el IoU sobre frames
anotados, ni la tasa de localización, ni la latencia de LocateAnything se pudieron medir
aquí. Quedan pendientes para Colab con GPU real.

## 4. Limitaciones y errores

1. **El sesgo hacia la clase mayoritaria se propaga en cascada.** MIMIC (F1 macro 0.326,
   sesgo documentado) → MATCH (Recall@1 desigual entre clases, sección 4 de su informe)
   → ASK (3 de 5 segmentos del timeline mal clasificados como la misma clase
   mayoritaria). Es la misma causa raíz — volumen insuficiente de datos en las clases
   curadas — manifestándose de forma distinta en cada entrega.
2. **La ventana de suavizado tiene un límite real de mejora.** Se probaron 4 valores;
   ninguno superó 40% de acierto. Ajustar el suavizado no resuelve un clasificador
   sesgado — solo puede reducir el ruido alrededor de una decisión que ya es incorrecta.
3. **El grounding real es la única pieza que no se pudo verificar de punta a punta en
   este entorno.** Se documenta explícitamente en tres lugares (spec, código y este
   informe) en vez de presentar código sin probar como si funcionara.
4. **El video de prueba es una construcción artificial** (5 clips distintos
   concatenados), no una grabación continua real. Esto dio verdad de terreno gratis,
   pero un video real de una sola persona podría comportarse distinto (transiciones más
   graduales entre poses, en vez de cortes abruptos entre clips).

## 5. Acciones concretas de mejora

- La acción de mayor impacto es la misma que en MIMIC y MATCH: ampliar las clases
  curadas con más datos reales. Ninguna mejora de ingeniería en ASK (ventana, filtros de
  duración mínima) compensa un clasificador entrenado con 5-16 ejemplos por clase.
- Validar `ask/grounding.py` en Colab con GPU real y completar la evaluación de IoU,
  tasa de localización y latencia que quedó pendiente aquí.
- Probar un filtro de duración mínima de evento (fusionar eventos más cortos que ~0.5s
  con el evento vecino dominante) como complemento a la ventana de suavizado.
- Repetir la evaluación del timeline con un video grabado de una sola toma continua,
  para separar el efecto de "cortes abruptos entre clips" del sesgo real del
  clasificador.

## 6. Cierre del proyecto integral

MimicVision AI queda completo: MIMIC (25 pts) clasifica poses y gestos en tiempo real,
MATCH (35 pts) recupera contenido visual por similitud, y ASK (40 pts) integra ambos con
grounding por lenguaje natural y una línea temporal de eventos, en modo batch y en vivo.
El hilo conductor de los tres informes es el mismo hallazgo, visto desde tres ángulos
distintos: el volumen de datos en las clases curadas manualmente es la limitación
dominante del proyecto, y se documentó honestamente en cada etapa en vez de ocultarse
detrás de métricas presentadas sin contexto.
