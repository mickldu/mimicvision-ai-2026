# MimicVision AI v0.1 — Informe de la Clase 1 (MIMIC)

Entrega: Clase 1 del proyecto integral MimicVision AI 2026.
Equipo: implementacion de referencia del docente.

## 1. Problema

Construir un sistema de vision por computador que reciba video desde webcam o archivo,
detecte landmarks del cuerpo, rostro y manos, transforme esos puntos en caracteristicas
geometricas y clasifique 10 categorias (5 posturas y 5 gestos) en tiempo real, mostrando
etiqueta, confianza y FPS.

## 2. Metodo

**Percepcion.** MediaPipe `HolisticLandmarker` (API de Tasks, mediapipe 0.10.35) extrae
33 puntos de pose, la malla facial y las manos en una sola pasada. Se eligio Holistic en
lugar de detectores separados porque varias clases dependen de la relacion entre partes
del cuerpo (mano-menton, mano-mano, manos-torso). Cada frame pasa por un lienzo fijo de
640x640 con escala uniforme antes de la deteccion; esto resuelve un defecto real de la
version 0.10.35 (crash con dimensiones variables entre frames consecutivos) sin
distorsionar la geometria.

**Caracteristicas.** Vector de 10 features geometricas calculadas solo con hombros,
codos, munecas, nariz y menton, normalizadas por el ancho de hombros: angulos de codo y
hombro (izquierdo y derecho), inclinacion de cabeza, distancia entre manos, distancia
mano-menton, distancia mano-hombro contrario, altura relativa de las manos y asimetria
entre manos. No se usan cadera ni rodilla porque las fotos de HaGRID son planos de medio
cuerpo y esas referencias no existen en 2 de las 10 clases.

**Datos.** Sin captura propia: el dataset combina HaGRID (CC-BY-SA 4.0; clases `saludo` y
`pulgar_arriba`, 120 imagenes cada una) con fotos Creative Commons curadas desde
Openverse para las otras 8 clases. Cada foto curada quedo registrada con URL de origen,
licencia y atribucion (`data/raw/cc_registro.csv`), paso un filtro automatico de
deteccion de torso y una revision visual con hojas de contacto donde se eliminaron
manualmente las imagenes que no correspondian a su clase (estatuas, pinturas, escenas
irrelevantes) y los duplicados.

**Clasificacion.** Se compararon SVM (kernel RBF) y Random Forest con validacion cruzada
estratificada; el modelo final se eligio por F1 macro, no accuracy, por el desbalance
entre las clases de HaGRID y las curadas. Split 70/15/15 fijado en `data/metadata.csv`
para que toda re-ejecucion use la misma particion.

**Tiempo real.** El pipeline aplica el modelo frame a frame con suavizado temporal por
ventana deslizante (mayoria movil) para estabilizar la etiqueta. Demo web con Gradio
(camara del navegador, identica en local y Colab) y demo de escritorio con OpenCV puro
para medir FPS sin latencia de red.

## 3. Resultados

(Los valores exactos se completan con la ejecucion del notebook `D1_mimic_baseline.ipynb`;
la matriz de confusion, el F1 por clase y la tabla de latencia/FPS quedan generados alli.)

- Modelo elegido y F1 macro de validacion: ver seccion 3 del notebook.
- Precision, recall y F1 por clase, y matriz de confusion: seccion 4 del notebook.
- Latencia media, p95 y FPS estimados: seccion 7 del notebook.
- Tasa de deteccion de landmarks por clase: seccion 2 del notebook.

## 4. Limitaciones y errores

1. **Volumen de datos.** Las 8 clases curadas quedaron por debajo del minimo recomendado
   de 100 muestras por clase (entre 10 y 40 tras la depuracion visual). Fue una decision
   deliberada: solo fotos reales verificadas, sin aumento de datos. El efecto esperado es
   mayor varianza en las metricas de esas clases y F1 mas debil que en las dos clases de
   HaGRID.
2. **Solapamiento semantico real.** `pensando` y `mano_en_menton` comparten geometria
   (mano cerca de la cara); `manos_juntas` y `zen` tambien se acercan cuando la persona
   medita con las manos unidas. La matriz de confusion del notebook permite cuantificar
   exactamente cuanto se confunden.
3. **Dominio de las fuentes.** HaGRID son selfies de interiores con la mano protagonista;
   las fotos CC son mas variadas (exteriores, blanco y negro, epocas distintas). El
   clasificador puede aprender el estilo de la fuente ademas del gesto — un riesgo tipico
   cuando cada clase viene de una fuente distinta, que se discute en clase como leccion.
4. **Deteccion imperfecta.** En fotos con personas pequenas o de espaldas el detector
   falla o produce landmarks pobres; la tasa de deteccion por clase queda registrada en el
   notebook y las muestras sin deteccion se excluyen del entrenamiento.
5. **Analisis de errores individuales.** El notebook (seccion 6) muestra hasta 5 errores
   reales del conjunto de test con su imagen, clase real y prediccion, para discusion.

## 5. Acciones concretas de mejora

- Ampliar las clases debiles con mas consultas curadas o con captura propia consentida.
- Anadir features de manos (dedos) desde los hand landmarks de Holistic para separar
  mejor `senalamiento` de `saludo`.
- Balancear el entrenamiento con pesos por clase en el SVM/RF.
- Medir la sensibilidad del modelo al estilo de la fuente entrenando con una fuente y
  evaluando con la otra.
