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

Ejecucion completa del notebook sobre el dataset final (330 imagenes registradas en
`data/metadata.csv`, 275 con landmarks utiles — 83.3% de tasa de deteccion global).

- **Modelo elegido:** Random Forest (le gano a SVM en la validacion cruzada).
- **F1 macro en validacion cruzada:** 0.326.
- **Accuracy en test:** 0.62. **F1 macro en test:** 0.30 (39 muestras de test en total).
- **Por clase en test** (precision / recall / F1): `saludo` 0.73/0.79/0.76 (14 muestras),
  `pulgar_arriba` 0.65/0.79/0.71 (14), `mano_en_menton` 1.00/1.00/1.00 (1),
  `pensando` 1.00/0.33/0.50 (3); las 6 clases restantes (`neutral`, `zen`,
  `brazos_cruzados`, `brazos_abiertos`, `manos_juntas`, `senalamiento`) quedaron en 0.00
  porque su conjunto de test tiene 1-2 muestras — una sola prediccion equivocada hunde el
  F1 completo de la clase.
- **Latencia del pipeline completo** (deteccion + features + clasificacion, CPU): 30.7 ms
  media, 41.4 ms p95, equivalente a ~32.5 FPS estimados — sobra margen para tiempo real.
- **Tasa de deteccion de landmarks por clase:** las dos clases de HaGRID tuvieron mas
  fallos de lo esperado (`saludo` 24/120 = 20%, `pulgar_arriba` 18/120 = 15%), porque
  varios recortes de HaGRID muestran solo la mano sin hombros visibles. Las clases
  curadas de Openverse tuvieron entre 1 y 5 fallos cada una.

## 4. Limitaciones y errores

1. **Volumen de datos, la limitacion dominante.** Las 8 clases curadas quedaron con 7 a
   16 imagenes utiles cada una (muy por debajo del minimo recomendado de 100), por la
   decision deliberada de usar solo fotos reales verificadas, sin aumento de datos. El
   split 70/15/15 sobre esos volumenes deja conjuntos de test de 1 a 3 muestras por
   clase, que es la causa directa de que 6 de las 10 clases muestren F1 de 0.00 en test:
   no es que el modelo sea incapaz de reconocerlas, es que una unica muestra de test no
   alcanza para medir nada de forma confiable. La metrica que si es representativa es la
   de `saludo` y `pulgar_arriba` (14 muestras de test cada una, F1 ~0.71-0.76).
2. **F1 macro de validacion moderado (0.326).** Confirma el problema anterior desde otro
   angulo: promediar el desempeno entre clases con 84 muestras de entrenamiento y clases
   con 5-11 es inherentemente inestable.
3. **Sesgo sistematico hacia las dos clases grandes.** La matriz de confusion (seccion 4
   del notebook) muestra un patron claro: casi todos los errores de las 8 clases chicas
   caen en `pulgar_arriba` o `saludo` (por ejemplo, `brazos_cruzados`, `manos_juntas` y
   `senalamiento` se predicen como `pulgar_arriba`; `neutral` como `brazos_cruzados`).
   Esto no es solo ruido de un test pequeno: es el comportamiento esperable de un Random
   Forest cuando una clase tiene 84 muestras de entrenamiento y otra tiene 5 -- el modelo
   aprende a "apostar" por la clase mayoritaria cuando la señal es ambigua. Confirma que
   la prioridad de mejora es balancear el dataset, no ajustar hiperparametros.
4. **Solapamiento semantico real.** `pensando` y `mano_en_menton` comparten geometria
   (mano cerca de la cara); `manos_juntas` y `zen` tambien se acercan cuando la persona
   medita con las manos unidas. La matriz de confusion permite verlo directamente en los
   pocos casos donde el error no fue simplemente "hacia la clase grande".
5. **Dominio de las fuentes.** HaGRID son selfies de interiores con la mano protagonista;
   las fotos CC son mas variadas (exteriores, blanco y negro, epocas distintas). El
   clasificador puede estar aprendiendo el estilo fotografico de cada fuente ademas del
   gesto — un riesgo tipico cuando cada clase viene de una fuente distinta.
6. **Deteccion imperfecta incluso en las clases grandes.** El 15-20% de fallo de deteccion
   en las clases de HaGRID muestra que "tener muchas imagenes" no evita el problema de
   fondo: los recortes de mano sin hombros visibles no producen pose util, sin importar
   cuantos haya.
7. **Analisis de errores individuales.** El notebook (seccion 6) muestra 5 de los 15
   errores reales del conjunto de test con su imagen, clase real y prediccion.

## 5. Acciones concretas de mejora

1. **Prioridad principal: mas datos reales en las 8 clases curadas** (objetivo minimo 30
   por clase). Es la unica accion que ataca la causa raiz de los F1 en 0.00 -- con 1-3
   muestras de test, ningun ajuste de modelo o de features va a estabilizar la metrica.
2. **Reemplazar el split fijo por cross-validation completa** cuando el dataset crezca,
   en vez de un hold-out de test tan pequeno para las clases curadas.
3. Anadir features de manos (dedos) desde los hand landmarks de Holistic para separar
   mejor `senalamiento` de `saludo`, y ayudar a distinguir `pensando` de
   `mano_en_menton`.
4. Balancear el entrenamiento con pesos por clase en el SVM/RF para compensar el
   desbalance entre HaGRID (84-120 muestras) y las clases curadas (7-16).
5. Medir la sensibilidad del modelo al estilo de la fuente entrenando con una fuente y
  evaluando con la otra, para confirmar o descartar la hipotesis de la seccion 4.4.
6. Investigar por que 15-20% de las imagenes de HaGRID no producen pose util, y si
   conviene filtrar por un criterio de encuadre antes de sumarlas al dataset.
