# MimicVision AI v0.5 — Informe de la Clase 2 (MATCH)

Entrega: Clase 2 del proyecto integral MimicVision AI 2026.
Equipo: implementación de referencia del docente.

## 1. Problema

Recuperar contenido visual similar a una pose o gesto observado, comparando tres
representaciones distintas: descriptores clásicos (HOG + LBP), geometría de landmarks
heredada de MIMIC, y embeddings visuales modernos (SigLIP2). El sistema debe devolver
Top-5 resultados con score y latencia, y explicar mediante un ablation study qué
componente aporta realmente.

## 2. Método

**Galería y consultas.** Sin curar dataset nuevo: se reutiliza `data/metadata.csv` de
MIMIC (330 imágenes, 10 clases). Para que las tres rutas se evalúen sobre exactamente
las mismas imágenes, se restringe a las 275 muestras donde MIMIC logró extraer
landmarks (necesarias para la ruta B) — de ahí quedan **236 imágenes de galería**
(splits train+val) y **39 consultas de evaluación** (split test), sin fuga de datos
entre ambos conjuntos.

**Ruta A (clásica).** HOG + LBP sobre la imagen en escala de grises redimensionada a
128×128. Vector concatenado de 1790 dimensiones (1764 de HOG + 26 de LBP).

**Ruta B (geométrica).** Reutiliza directamente el vector de 10 features geométricas ya
calculado por MIMIC (`data/features_mimic.csv`) — cero recomputación.

**Ruta C (moderna).** Embeddings de SigLIP2 (`google/siglip2-base-patch16-224`,
Apache-2.0, sin gate — se descartó DINOv3 por requerir autenticación gated en Hugging
Face). Vector de 768 dimensiones.

**Índice.** Similitud coseno con numpy (sin FAISS: a 236 vectores una búsqueda exacta es
instantánea y evita una dependencia pesada innecesaria).

**Evaluación.** Recall@1, Recall@5 y MRR, con el criterio de relevancia "misma clase que
la consulta". Las 39 consultas se categorizan en fáciles (HaGRID: `saludo`,
`pulgar_arriba`), difíciles (clases curadas chicas) y ambiguas (`pensando`,
`mano_en_menton`, `zen`, `manos_juntas` — ya sabíamos por MIMIC que se confunden entre
sí).

## 3. Resultados

| Ruta | Recall@1 | Recall@5 | MRR | Latencia indexación (ms) | Latencia consulta (ms) |
|---|---|---|---|---|---|
| A: HOG+LBP | 0.538 | 0.744 | 0.623 | 7.3 | 0.55 |
| B: geometría | 0.436 | 0.795 | 0.569 | 0.2 | 0.15 |
| **C: SigLIP2** | **0.846** | **1.000** | **0.907** | 8.5 | 0.33 |

SigLIP2 gana con claridad en las tres métricas — Recall@5 perfecto (1.0) sobre las 39
consultas de test. La geometría (ruta B) es la más débil en Recall@1 pero, curiosamente,
supera a HOG+LBP en Recall@5 (0.795 vs 0.744): la geometría de pose ubica la clase
correcta *cerca* del top más seguido que HOG+LBP, aunque no siempre en el primer lugar.

**Por categoría de consulta (Recall@1):**

| Ruta | Fácil | Difícil | Ambigua |
|---|---|---|---|
| A: HOG+LBP | 0.71 | 0.25 | 0.00 |
| B: geometría | 0.54 | 0.00 | 0.29 |
| C: SigLIP2 | 0.96 | 0.50 | 0.57 |

SigLIP2 domina en las tres categorías, pero el patrón confirma lo esperado: todas las
rutas caen en las clases difíciles y ambiguas frente a las fáciles. Ninguna ruta
clásica (A o B) resuelve las clases ambiguas de forma aceptable (0.00 y 0.29).

## 4. Ablation study

### 4.1 Textura: HOG solo vs HOG + LBP

| Variante | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Solo HOG | 0.5385 | 0.7436 | 0.6227 |
| HOG + LBP | 0.5385 | 0.7436 | 0.6227 |

**Resultado inesperado y real: son idénticos, bit a bit.** La causa no es un error de
código sino de escala: LBP aporta solo 26 dimensiones normalizadas frente a las 1764 de
HOG. Al concatenar sin ponderar y calcular similitud coseno, el descriptor de textura
queda diluido en menos del 1.5% del vector — su contribución a la dirección del vector
final es prácticamente nula. Lección concreta: agregar un descriptor no sirve de nada
si su escala/dimensionalidad no se pondera respecto al resto del vector.

### 4.2 Crop de persona vs frame completo (ruta C)

| Variante | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Frame completo | 0.846 | 1.000 | 0.907 |
| Crop de persona | 0.795 | 0.923 | 0.841 |

**El crop empeora el resultado**, en las tres métricas. Hipótesis: SigLIP2 está
aprovechando también el contexto de la escena completa (fondo, encuadre), y ese contexto
correlaciona con la fuente del dataset (los recortes de HaGRID vienen de selfies de
interior; las fotos CC curadas son más variadas). Recortar a la persona elimina esa
señal "fácil" de fondo y obliga al modelo a apoyarse más en la pose en sí — lo cual baja
el Recall porque el fondo, aunque no es la señal que se quiere medir, ayudaba a acertar.
Esto conecta directamente con la limitación de "dominio de las fuentes" ya señalada en
el informe de MIMIC.

### 4.3 Rutas individuales

Ya cubierto en la sección 3 — SigLIP2 > HOG+LBP > geometría en Recall@1.

### 4.4 Re-ranking geométrico (experimento E-D)

| Variante | Recall@1 |
|---|---|
| SigLIP2 sin re-ranking | 0.846 |
| SigLIP2 + re-ranking por geometría | 0.769 |

**El re-ranking también empeora el resultado.** Es el resultado esperable dado que la
ruta B (geometría) tiene, por sí sola, el Recall@1 más bajo de las tres (0.436):
reordenar el Top-10 de la ruta ganadora usando la señal más débil del experimento
introduce ruido en vez de mejorar la precisión. La combinación de señales solo ayuda
cuando la señal secundaria aporta información complementaria y de calidad comparable —
no es el caso aquí.

## 5. Limitaciones y errores

1. **Las tres rutas se evalúan sobre solo 39 consultas.** Con clases curadas de 5-16
   imágenes en total, cada clase aporta 1-3 consultas de test — el mismo problema de
   volumen de datos ya documentado en el informe de MIMIC se propaga a MATCH.
2. **El ablation de textura revela un problema metodológico de diseño, no de datos:**
   concatenar descriptores de dimensionalidad muy distinta sin normalizar/ponderar por
   separado anula al más chico. Se explica en la sección 4.1 en vez de ocultarlo.
3. **Los dos ablation restantes (crop y re-ranking) dieron resultados negativos.** Se
   reportan igual que los positivos: el objetivo del ablation es medir el efecto real de
   cada componente, no confirmar una hipótesis favorable.
4. **Posible fuga de señal por el fondo de la imagen.** El hallazgo de la sección 4.2
   sugiere que parte del Recall@1 de SigLIP2 podría depender del estilo fotográfico de
   cada fuente (HaGRID vs Openverse) y no solo de la pose/gesto — coherente con la
   limitación de "dominio de las fuentes" del informe de MIMIC.

## 6. Acciones concretas de mejora

- Ponderar o normalizar por separado HOG y LBP antes de concatenar (o combinarlos con un
  score de fusión en vez de una concatenación cruda) para que el ablation de textura
  mida algo real.
- Repetir el ablation de crop de persona separando por fuente (HaGRID vs Openverse) para
  confirmar o descartar la hipótesis de fuga de señal por el fondo.
- Probar re-ranking con pesos (combinar el score de SigLIP2 y el de geometría en vez de
  reemplazar el orden por completo), ya que el re-ranking puro con una señal más débil
  perjudicó el resultado.
- Ampliar las clases curadas (misma acción prioritaria que en MIMIC) para tener más de
  1-3 consultas de test por clase y metricas menos ruidosas por categoría.
