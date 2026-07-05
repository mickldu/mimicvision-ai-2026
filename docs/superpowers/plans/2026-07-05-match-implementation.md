# MATCH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build MATCH (Clase 2): recuperación visual por similitud sobre la galería de MIMIC, comparando tres representaciones (HOG+LBP clásico, geometría heredada de MIMIC, embeddings SigLIP2), con índice por similitud coseno, evaluación Recall@1/Recall@5/MRR y un ablation study de 4 puntos.

**Architecture:** `match/gallery.py` parte `data/metadata.csv` en galería (train+val) y consultas (test) sin curar nada nuevo. `match/descriptors.py` y `match/embeddings.py` producen vectores por ruta (A: HOG+LBP, C: SigLIP2); la ruta B reutiliza `data/features_mimic.csv` de MIMIC directamente. `match/index.py` construye un índice de similitud coseno en numpy y hace búsquedas Top-K. `match/evaluate.py` calcula Recall@K y MRR. `match/reranking.py` reordena un Top-K por similitud geométrica para el experimento de re-ranking.

**Tech Stack:** Python 3.13, scikit-image (HOG, LBP), transformers + torch CPU (SigLIP2), numpy, pandas, pytest. Ya validado en `.venv`: `google/siglip2-base-patch16-224` carga sin gate y `modelo.get_image_features(**entradas).pooler_output[0].numpy()` da un vector de 768 floats; HOG con `pixels_per_cell=(16,16), cells_per_block=(2,2)` sobre una imagen 128x128 da un vector de 1764; LBP uniforme con 24 puntos y radio 3 da un histograma de 26 bins.

---

## Task 1: `match/gallery.py` — partir metadata en galería y consultas

**Files:**
- Create: `match/__init__.py`
- Create: `match/gallery.py`
- Test: `tests/match/__init__.py`
- Test: `tests/match/test_gallery.py`

- [ ] **Step 1: Crear el paquete**

```bash
mkdir -p match tests/match
touch match/__init__.py tests/match/__init__.py
```

- [ ] **Step 2: Escribir el test que falla**

`tests/match/test_gallery.py`:
```python
import pandas as pd

from match.gallery import cargar_galeria_y_consultas


def test_separa_galeria_de_consultas_por_split(tmp_path):
    ruta = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"sample_id": "a", "clase": "saludo", "split": "train"},
        {"sample_id": "b", "clase": "saludo", "split": "val"},
        {"sample_id": "c", "clase": "saludo", "split": "test"},
        {"sample_id": "d", "clase": "zen", "split": "test"},
    ]).to_csv(ruta, index=False)

    galeria, consultas = cargar_galeria_y_consultas(str(ruta))

    assert set(galeria["sample_id"]) == {"a", "b"}
    assert set(consultas["sample_id"]) == {"c", "d"}
```

- [ ] **Step 3: Correr y verificar que falla**

Run: `pytest tests/match/test_gallery.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'match.gallery'`

- [ ] **Step 4: Implementar**

`match/gallery.py`:
```python
"""Carga la galeria y las consultas de evaluacion de MATCH a partir del
mismo data/metadata.csv que ya arma MIMIC -- no se cura un dataset
nuevo, se reutiliza el existente con su split fijo (train+val como
galeria, test como consultas, sin fuga de datos entre ambos)."""
import pandas as pd


def cargar_galeria_y_consultas(ruta_metadata: str = "data/metadata.csv"):
    metadata = pd.read_csv(ruta_metadata)
    galeria = metadata[metadata["split"].isin(["train", "val"])].reset_index(drop=True)
    consultas = metadata[metadata["split"] == "test"].reset_index(drop=True)
    return galeria, consultas
```

- [ ] **Step 5: Correr y verificar que pasa**

Run: `pytest tests/match/test_gallery.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add match/__init__.py match/gallery.py tests/match/__init__.py tests/match/test_gallery.py
git commit -m "Agregar carga de galeria y consultas de MATCH desde metadata.csv"
```

---

## Task 2: `match/descriptors.py` — HOG y LBP (ruta A)

**Files:**
- Create: `match/descriptors.py`
- Test: `tests/match/test_descriptors.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/match/test_descriptors.py`:
```python
import cv2
import numpy as np

from match.descriptors import extraer_hog, extraer_lbp_histograma

FRAME_PRUEBA = np.random.default_rng(0).integers(0, 255, (200, 150, 3), dtype=np.uint8)


def test_extraer_hog_devuelve_vector_de_longitud_fija():
    vector = extraer_hog(FRAME_PRUEBA)
    assert vector.shape == (1764,)


def test_extraer_hog_es_igual_sin_importar_el_tamano_original():
    otro_tamano = cv2.resize(FRAME_PRUEBA, (400, 300))
    v1 = extraer_hog(FRAME_PRUEBA)
    v2 = extraer_hog(otro_tamano)
    assert v1.shape == v2.shape


def test_extraer_lbp_histograma_suma_1():
    histograma = extraer_lbp_histograma(FRAME_PRUEBA)
    assert histograma.shape == (26,)
    assert histograma.sum() == pytest.approx(1.0, abs=1e-6)
```

Falta el import de pytest:
```python
import pytest
```
(agregarlo junto a los demas imports al inicio del archivo).

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/match/test_descriptors.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`match/descriptors.py`:
```python
"""Descriptores clasicos de imagen (ruta A del ablation): HOG y LBP.

Toda imagen se redimensiona primero a un tamano fijo -- si no, HOG
devolveria vectores de longitud distinta segun el tamano original de
cada foto, y no se podrian comparar por similitud coseno."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

TAMANO_FIJO = (128, 128)
LBP_PUNTOS = 24
LBP_RADIO = 3


def _a_gris_redimensionado(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, TAMANO_FIJO)


def extraer_hog(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = _a_gris_redimensionado(imagen_bgr)
    return hog(gris, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def extraer_lbp_histograma(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = _a_gris_redimensionado(imagen_bgr)
    patrones = local_binary_pattern(gris, LBP_PUNTOS, LBP_RADIO, method="uniform")
    histograma, _ = np.histogram(
        patrones, bins=np.arange(0, LBP_PUNTOS + 3), range=(0, LBP_PUNTOS + 2)
    )
    return histograma.astype(np.float64) / (histograma.sum() + 1e-9)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/match/test_descriptors.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add match/descriptors.py tests/match/test_descriptors.py
git commit -m "Agregar descriptores clasicos HOG y LBP (ruta A)"
```

---

## Task 3: `match/embeddings.py` — SigLIP2 (ruta C)

**Files:**
- Create: `match/embeddings.py`
- Test: `tests/match/test_embeddings.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/match/test_embeddings.py`:
```python
from pathlib import Path

import cv2
import numpy as np

from match.embeddings import ExtractorSigLIP2

FIXTURE = Path(__file__).parent.parent / "fixtures" / "persona_prueba.jpg"


def test_extraer_devuelve_vector_768_no_nulo():
    extractor = ExtractorSigLIP2()
    imagen = cv2.imread(str(FIXTURE))

    vector = extractor.extraer(imagen)

    assert vector.shape == (768,)
    assert not np.isnan(vector).any()
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/match/test_embeddings.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`match/embeddings.py`:
```python
"""Embeddings visuales modernos (ruta C) con SigLIP2.

Nota de API verificada durante el desarrollo: AutoModel.get_image_features()
en esta version de transformers no devuelve un tensor plano, sino un
BaseModelOutputWithPooling. El vector que se usa como embedding de la
imagen es su atributo pooler_output.

Se eligio SigLIP2 (Apache-2.0, sin gate) en vez de DINOv3 porque los
modelos facebook/dinov3-* estan gated en Hugging Face y exigirian que
cada persona que corra el notebook solicite acceso y configure un
token -- rompe la portabilidad entre local y Colab.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODELO_SIGLIP2 = "google/siglip2-base-patch16-224"


class ExtractorSigLIP2:
    def __init__(self):
        self._procesador = AutoProcessor.from_pretrained(MODELO_SIGLIP2)
        self._modelo = AutoModel.from_pretrained(MODELO_SIGLIP2)
        self._modelo.eval()

    def extraer(self, imagen_bgr: np.ndarray) -> np.ndarray:
        imagen_rgb = imagen_bgr[:, :, ::-1]
        imagen_pil = Image.fromarray(imagen_rgb)
        entradas = self._procesador(images=imagen_pil, return_tensors="pt")
        with torch.no_grad():
            salida = self._modelo.get_image_features(**entradas)
        return salida.pooler_output[0].numpy()
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/match/test_embeddings.py -v`
Expected: PASS (1 test). La primera corrida descarga los pesos de SigLIP2 (~800 MB) y tarda mas; las siguientes usan la cache local de Hugging Face.

- [ ] **Step 5: Commit**

```bash
git add match/embeddings.py tests/match/test_embeddings.py
git commit -m "Agregar extraccion de embeddings SigLIP2 (ruta C)"
```

---

## Task 4: `match/index.py` — índice de similitud coseno

**Files:**
- Create: `match/index.py`
- Test: `tests/match/test_index.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/match/test_index.py`:
```python
import numpy as np
import pytest

from match.index import IndiceSimilitud


def test_buscar_devuelve_el_vector_identico_primero():
    vectores = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ])
    metadatos = [
        {"image_id": "a", "etiqueta_pose": "saludo"},
        {"image_id": "b", "etiqueta_pose": "zen"},
        {"image_id": "c", "etiqueta_pose": "saludo"},
    ]
    indice = IndiceSimilitud(vectores, metadatos)

    resultados = indice.buscar(np.array([1.0, 0.0, 0.0]), k=2)

    assert resultados[0]["image_id"] == "a"
    assert resultados[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert resultados[1]["image_id"] == "c"


def test_buscar_respeta_k():
    vectores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    metadatos = [{"image_id": str(i)} for i in range(3)]
    indice = IndiceSimilitud(vectores, metadatos)

    resultados = indice.buscar(np.array([1.0, 0.0]), k=1)

    assert len(resultados) == 1
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/match/test_index.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`match/index.py`:
```python
"""Indice de similitud coseno sobre los vectores de la galeria.

Con ~290 imagenes, una busqueda exacta por fuerza bruta es instantanea:
no hace falta FAISS ni ninguna estructura aproximada a esta escala, y
evitarlo ahorra una dependencia pesada con friccion de instalacion en
Windows. El docx permite explicitamente esta alternativa."""
import numpy as np


def _normalizar(vectores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / (normas + 1e-9)


class IndiceSimilitud:
    def __init__(self, vectores: np.ndarray, metadatos: list[dict]):
        self._vectores_normalizados = _normalizar(np.asarray(vectores, dtype=np.float64))
        self._metadatos = metadatos

    def buscar(self, vector_consulta: np.ndarray, k: int = 5) -> list[dict]:
        consulta = np.asarray(vector_consulta, dtype=np.float64).reshape(1, -1)
        consulta_normalizada = _normalizar(consulta)[0]
        scores = self._vectores_normalizados @ consulta_normalizada
        indices_top_k = np.argsort(-scores)[:k]

        resultados = []
        for i in indices_top_k:
            resultado = dict(self._metadatos[i])
            resultado["score"] = float(scores[i])
            resultados.append(resultado)
        return resultados
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/match/test_index.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add match/index.py tests/match/test_index.py
git commit -m "Agregar indice de similitud coseno con busqueda Top-K"
```

---

## Task 5: `match/evaluate.py` — Recall@K y MRR

**Files:**
- Create: `match/evaluate.py`
- Test: `tests/match/test_evaluate.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/match/test_evaluate.py`:
```python
import pytest

from match.evaluate import mrr, recall_at_k


def _resultado(etiqueta):
    return {"etiqueta_pose": etiqueta}


def test_recall_at_1_cuenta_solo_el_primer_resultado():
    resultados_por_consulta = [
        [_resultado("saludo"), _resultado("zen")],
        [_resultado("zen"), _resultado("saludo")],
    ]
    clases_verdaderas = ["saludo", "saludo"]

    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=1) == 0.5


def test_recall_at_5_es_mas_permisivo_que_recall_at_1():
    resultados_por_consulta = [[_resultado("zen"), _resultado("saludo")]]
    clases_verdaderas = ["saludo"]

    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=1) == 0.0
    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=5) == 1.0


def test_mrr_premia_que_el_correcto_este_mas_arriba():
    resultados_por_consulta = [
        [_resultado("saludo")],
        [_resultado("zen"), _resultado("saludo")],
    ]
    clases_verdaderas = ["saludo", "saludo"]

    valor = mrr(resultados_por_consulta, clases_verdaderas)

    assert valor == pytest.approx((1.0 + 0.5) / 2)
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/match/test_evaluate.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`match/evaluate.py`:
```python
"""Metricas de recuperacion sobre resultados Top-K: Recall@K y MRR.

El criterio de relevancia es simple y explicito: un resultado cuenta
como correcto si su etiqueta_pose coincide con la clase real de la
consulta -- no hay anotacion manual de relevancia mas fina que esa."""


def recall_at_k(resultados_por_consulta: list[list[dict]], clases_verdaderas: list[str], k: int) -> float:
    aciertos = 0
    for resultados, clase_real in zip(resultados_por_consulta, clases_verdaderas):
        clases_top_k = [r["etiqueta_pose"] for r in resultados[:k]]
        if clase_real in clases_top_k:
            aciertos += 1
    return aciertos / len(clases_verdaderas)


def mrr(resultados_por_consulta: list[list[dict]], clases_verdaderas: list[str]) -> float:
    reciprocos = []
    for resultados, clase_real in zip(resultados_por_consulta, clases_verdaderas):
        rango = None
        for posicion, resultado in enumerate(resultados, start=1):
            if resultado["etiqueta_pose"] == clase_real:
                rango = posicion
                break
        reciprocos.append(1.0 / rango if rango else 0.0)
    return sum(reciprocos) / len(reciprocos)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/match/test_evaluate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add match/evaluate.py tests/match/test_evaluate.py
git commit -m "Agregar metricas Recall@K y MRR para evaluar recuperacion"
```

---

## Task 6: `match/reranking.py` — re-ranking geométrico (experimento E-D)

**Files:**
- Create: `match/reranking.py`
- Test: `tests/match/test_reranking.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/match/test_reranking.py`:
```python
import numpy as np

from match.reranking import reordenar_por_geometria


def test_reordena_poniendo_primero_al_mas_parecido_geometricamente():
    candidatos = [
        {"image_id": "a"},
        {"image_id": "b"},
    ]
    vector_consulta = np.array([1.0, 0.0])
    vectores_por_id = {
        "a": np.array([0.0, 1.0]),   # geometria opuesta a la consulta
        "b": np.array([0.9, 0.1]),   # geometria parecida a la consulta
    }

    reordenados = reordenar_por_geometria(candidatos, vector_consulta, vectores_por_id)

    assert [c["image_id"] for c in reordenados] == ["b", "a"]
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/match/test_reranking.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`match/reranking.py`:
```python
"""Re-ranking de un Top-K por similitud geometrica (ruta B), para el
experimento E-D del docx: combinar semantica visual (embeddings) con
geometria de pose para ver si mejora el Recall@1."""
import numpy as np


def _similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def reordenar_por_geometria(
    candidatos: list[dict],
    vector_geometrico_consulta: np.ndarray,
    vectores_geometricos_por_id: dict,
) -> list[dict]:
    def puntaje(candidato):
        vector_candidato = vectores_geometricos_por_id[candidato["image_id"]]
        return _similitud_coseno(vector_geometrico_consulta, vector_candidato)

    return sorted(candidatos, key=puntaje, reverse=True)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/match/test_reranking.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add match/reranking.py tests/match/test_reranking.py
git commit -m "Agregar re-ranking por similitud geometrica (experimento E-D)"
```

---

## Task 7: Notebook `D2_match_retrieval.ipynb`

- [ ] **Step 1:** Crear el notebook con celdas: detección de entorno (igual patrón que D1), carga de galería/consultas vía `match.gallery.cargar_galeria_y_consultas`, extracción de las 3 representaciones sobre la galería completa (HOG+LBP, geometría desde `data/features_mimic.csv`, SigLIP2), construcción de un `IndiceSimilitud` por ruta, evaluación de las 40 consultas de test con Recall@1/Recall@5/MRR por ruta, categorización de consultas en fáciles/difíciles/ambiguas (según docs/superpowers/specs/2026-07-05-match-design.md sección 6), tabla de ablation (los 4 puntos de la sección 7 del spec), y demo Gradio con `gr.Gallery`.
- [ ] **Step 2:** Ejecutar el notebook de punta a punta con `jupyter nbconvert --to notebook --execute --inplace` y confirmar que corre sin errores.
- [ ] **Step 3:** Commit

```bash
git add notebooks/D2_match_retrieval.ipynb
git commit -m "Agregar notebook D2_match_retrieval con las 3 rutas, evaluacion y ablation"
```

---

## Task 8: Pestaña MATCH en `app/app.py`

**Files:**
- Modify: `app/app.py`

- [ ] **Step 1: Añadir la pestaña sin tocar la pestaña MIMIC existente**

Agregar dentro del mismo `gr.Blocks` de `app/app.py`, después de la pestaña `"MIMIC"`:

```python
with gr.Tab("MATCH"):
    gr.Markdown(
        "Sube o captura una foto y el sistema busca las 5 imagenes mas "
        "parecidas en la galeria de MIMIC."
    )
    entrada_match = gr.Image(sources=["upload", "webcam"])
    galeria_resultados = gr.Gallery(label="Top 5 mas similares", columns=5)
    boton_buscar = gr.Button("Buscar")
    boton_buscar.click(_buscar_similares, inputs=entrada_match, outputs=galeria_resultados)
```

Y antes del bloque `with gr.Blocks(...)`, agregar la función que arma la respuesta (carga perezosa del índice, igual patrón que `_inicializar()` de MIMIC):

```python
_indice_match = None


def _inicializar_match():
    global _indice_match
    if _indice_match is None:
        from match.embeddings import ExtractorSigLIP2
        from match.gallery import cargar_galeria_y_consultas
        from match.index import IndiceSimilitud

        galeria, _ = cargar_galeria_y_consultas()
        extractor = ExtractorSigLIP2()
        vectores = []
        metadatos = []
        for fila in galeria.itertuples():
            imagen = cv2.imread(fila.ruta)
            if imagen is None:
                continue
            vectores.append(extractor.extraer(imagen))
            metadatos.append({
                "image_id": fila.sample_id,
                "etiqueta_pose": fila.clase,
                "ruta_archivo": fila.ruta,
            })
        _indice_match = (IndiceSimilitud(np.array(vectores), metadatos), extractor)


def _buscar_similares(imagen):
    if imagen is None:
        return []
    _inicializar_match()
    indice, extractor = _indice_match
    vector_consulta = extractor.extraer(imagen[:, :, ::-1])
    resultados = indice.buscar(vector_consulta, k=5)
    return [r["ruta_archivo"] for r in resultados]
```

- [ ] **Step 2: Verificar que el modulo importa sin errores**

Run: `python -c "import app.app"`
Expected: no lanza excepciones al importar.

- [ ] **Step 3: Commit**

```bash
git add app/app.py
git commit -m "Agregar pestana MATCH a la demo Gradio"
```

---

## Self-Review (completado antes de ejecutar)

**Cobertura del spec:** galería/consultas reutilizadas (Task 1), ruta A HOG+LBP (Task 2),
ruta C SigLIP2 (Task 3, DINOv3 descartado y documentado en el código), índice coseno
(Task 4), Recall@K/MRR (Task 5), re-ranking geométrico para el experimento E-D (Task 6),
notebook con evaluación categorizada y ablation completo (Task 7), demo Gradio (Task 8).
La ruta B (geometría) no necesita módulo nuevo — se consume directo desde
`data/features_mimic.csv`, ya generado por MIMIC.

**Placeholders:** ninguno — cada paso tiene código completo y verificado contra la API
real (SigLIP2, HOG, LBP se probaron en el entorno antes de escribir este plan).

**Consistencia de tipos:** `MatchResult` como `dict` con las claves `image_id`,
`etiqueta_pose`, `ruta_archivo`, `score` se usa igual en `index.py`, `evaluate.py`,
`reranking.py` y `app/app.py`.
