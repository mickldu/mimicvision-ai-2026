# MATCH (Clase 2) — diseño detallado

Fecha: 2026-07-05
Estado: aprobado por el usuario (docente), pendiente de revisión final del documento escrito.
Depende de: `2026-07-05-mimicvision-overview-design.md` (arquitectura compartida) y
`2026-07-05-mimic-design.md` (Clase 1, ya implementada — MIMIC v0.1).

## 1. Alcance de este sub-proyecto

Segunda entrega del proyecto integral MimicVision AI (35 pts). Recupera contenido visual
similar a una pose o gesto observado, comparando tres representaciones: descriptores
clásicos (HOG + LBP), geometría de landmarks (heredada de MIMIC) y embeddings visuales
modernos (SigLIP2). El alcance completo (objetivos, actividades obligatorias, criterios
de aceptación, rúbrica) está en
`docs/Proyecto_Integral_MimicVision_Clase_2_MATCH.docx`; aquí solo se documentan las
decisiones de diseño específicas de esta implementación de referencia.

## 2. Fix retroactivo a MIMIC: `bbox_persona`

El contrato `PerceptionResult` (definido en el diseño transversal) promete un
`bbox_persona`, pero la implementación de MIMIC v0.1 lo dejó siempre en `None` — nunca se
calculó. MATCH necesita ese campo para su ablation de "crop de persona vs frame completo",
así que antes de construir MATCH se completa `mimic/pipeline.py` (y el vector que ya
existe en `mimic/features.py`): el bounding box se calcula como el rectángulo que encierra
todos los landmarks de pose usados en las features (hombros, codos, muñecas, nariz), con
un margen del 15% en cada lado, convertido a coordenadas de píxel con las dimensiones del
frame. Esto no cambia ninguna de las 10 features geométricas ni los modelos ya
entrenados — solo completa un campo que estaba pendiente.

## 3. Galería y separación de datos

Se reutiliza el dataset de MIMIC sin curar nada nuevo:

- **Galería:** splits `train` + `val` de `data/metadata.csv` (~290 imágenes, 10 clases).
- **Consultas de evaluación:** split `test` (~40 imágenes) — ya separado de la galería
  desde la Clase 1, por lo que no hay fuga de datos entre índice y evaluación.

Esto satisface de sobra el mínimo del docx (≥300 imágenes, ≥6 categorías) sin trabajo de
curación adicional, y es coherente con "la Clase 2 reutiliza el pipeline de la Clase 1".

## 4. Tres rutas de representación

| Ruta | Representación | Módulo |
|---|---|---|
| A | HOG + LBP (descriptores clásicos, scikit-image) | `match/descriptors.py` |
| B | Vector geométrico de 10 features (heredado de MIMIC) | reutiliza `data/features_mimic.csv` |
| C | Embedding SigLIP2 (`google/siglip2-base-patch16-224`, Apache-2.0, sin gate) | `match/embeddings.py` |

Se descartó DINOv3 (sugerido como alternativa en el docx) porque los modelos
`facebook/dinov3-*` están *gated* en Hugging Face: exigen solicitar acceso y autenticarse
con un token por cada usuario, lo cual rompe "debe funcionar igual en local y en Colab"
para cualquiera que clone el repositorio sin esa aprobación previa. SigLIP2 es la primera
opción que el propio docx sugiere y no tiene esa fricción.

## 5. Índice y búsqueda

Similitud coseno calculada directamente con numpy (`match/index.py`), no FAISS. Con ~290
vectores en la galería, un índice aproximado no aporta nada frente a una búsqueda exacta
por fuerza bruta, y evita sumar una dependencia pesada con fricción de instalación en
Windows. El docx permite explícitamente esta alternativa ("FAISS o cosine_similarity").
Cada consulta devuelve Top-5 con score de similitud y latencia medida.

## 6. Evaluación

Métricas por ruta (A, B, C): **Recall@1**, **Recall@5** y **MRR**, usando como criterio de
relevancia que la imagen recuperada pertenezca a la misma clase que la consulta. Se mide
también la **latencia de indexación** (una vez, sobre toda la galería) y la **latencia de
consulta** (por búsqueda), por ruta.

Las 40 consultas de test se agrupan en tres categorías para el análisis:

- **Fáciles:** clases de HaGRID (`saludo`, `pulgar_arriba`) — mucho soporte en galería.
- **Difíciles:** clases curadas chicas (7-16 imágenes en galería cada una).
- **Ambiguas:** pares que la matriz de confusión de MIMIC ya mostró que se confunden
  (`pensando`/`mano_en_menton`, `zen`/`manos_juntas`).

Esta categorización explica los resultados en vez de solo reportarlos.

## 7. Ablation study

Se retira o modifica un componente a la vez, midiendo su efecto sobre Recall@1:

1. **Textura:** HOG solo vs HOG + LBP.
2. **Crop de persona:** SigLIP2 sobre el frame completo vs sobre el crop de persona
   (usa el `bbox_persona` corregido en la sección 2).
3. **Rutas individuales:** A vs B vs C, para ver cuál generaliza mejor a las clases
   curadas chicas frente a las clases grandes de HaGRID.
4. **Re-ranking (experimento E-D del docx):** la mejor ruta individual entrega un Top-10,
   que se re-ordena por similitud geométrica (ruta B) — mide si combinar semántica visual
   con geometría de pose mejora el Recall@1 final.

## 8. Estructura de módulos

```
match/
  gallery.py         # carga galeria+metadata desde data/metadata.csv (train+val)
  descriptors.py       # HOG + LBP (ruta A)
  embeddings.py          # SigLIP2 (ruta C)
  index.py                 # similitud coseno, build() y search() -> MatchResult[]
  evaluate.py                 # Recall@1, Recall@5, MRR, latencia, ablation
```

`match/index.py` produce `MatchResult` (definido en el diseño transversal): `image_id`,
`timestamp` (None para imágenes fijas de galería), `embedding_id`, `etiqueta_pose`,
`ruta_archivo`, `score`.

## 9. Notebook y entregables

`D2_match_retrieval.ipynb`: detección de entorno (igual que D1), carga de galería y
consultas desde `data/metadata.csv`, extracción de las 3 representaciones, construcción
del índice, evaluación por ruta con la categorización fácil/difícil/ambigua, tabla de
ablation, y demo Gradio (pestaña MATCH: sube o captura una imagen, muestra Top-5 con
score y latencia). Reutiliza el mismo patrón de comentarios en español humanizados que
D1.

## 10. Interfaz Gradio

Se añade una pestaña "MATCH" a `app/app.py` (no se toca la pestaña MIMIC existente):
`gr.Image(sources=["upload", "webcam"])` para la consulta, y una galería de resultados
(`gr.Gallery`) mostrando las 5 imágenes más similares con su score.
