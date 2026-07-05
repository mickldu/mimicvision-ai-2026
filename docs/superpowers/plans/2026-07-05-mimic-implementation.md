# MIMIC (Clase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working module of MimicVision AI: real-time classification of 5 body poses + 5 hand gestures from webcam/video, with a trained classical classifier, a Gradio demo, and a Jupyter notebook that runs identically in Colab and locally.

**Architecture:** MediaPipe Tasks `HolisticLandmarker` extracts pose/hand/face landmarks per frame → `mimic/features.py` converts landmarks into a 10-dimensional geometric feature vector (angles/distances normalized by shoulder width) → `mimic/classifier.py` trains and compares SVM vs Random Forest, picks the best by macro F1 → `mimic/temporal.py` smooths per-frame predictions → `mimic/pipeline.py` wires it all into a single `PerceptionResult` used by the notebook, the live demo script, and the Gradio app.

**Tech Stack:** Python 3.13, mediapipe 0.10.35 (Tasks API), opencv-python, scikit-learn, pandas, gradio, pytest. Already validated locally in `.venv` (see commits from the design phase — HolisticLandmarker confirmed working against `mediapipe.tasks.python.vision.HolisticLandmarker`, model file downloaded to `models/holistic_landmarker.task`).

---

## Task 1: Scaffolding del proyecto y fixture de prueba

**Files:**
- Create: `mimic/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/mimic/__init__.py`
- Create: `tests/fixtures/generar_fixture.py`
- Create: `tests/fixtures/persona_prueba.jpg` (generado por el script anterior)
- Create: `pytest.ini`

- [ ] **Step 1: Crear carpetas y `__init__.py` vacíos**

```bash
mkdir -p mimic tests/mimic tests/fixtures
touch mimic/__init__.py tests/__init__.py tests/mimic/__init__.py
```

- [ ] **Step 2: Generar una imagen de prueba real y reproducible (sin depender de internet en cada test run)**

`tests/fixtures/generar_fixture.py`:
```python
"""Genera una foto de prueba real (no sintetica) a partir del dataset
de ejemplo que trae scikit-image, para que los tests de landmarks
tengan una imagen de persona reproducible sin depender de internet."""
from pathlib import Path
import cv2
from skimage import data

def generar():
    imagen = data.astronaut()
    ruta = Path(__file__).parent / "persona_prueba.jpg"
    cv2.imwrite(str(ruta), cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
    return ruta

if __name__ == "__main__":
    print(generar())
```

Run: `cd mimicvision-ai && source .venv/Scripts/activate && python tests/fixtures/generar_fixture.py`
Expected: imprime la ruta `tests/fixtures/persona_prueba.jpg` y el archivo existe.

- [ ] **Step 3: Configurar pytest**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: Commit**

```bash
git add mimic/__init__.py tests/ pytest.ini
git commit -m "Agregar esqueleto del paquete mimic y fixture de prueba real"
```

---

## Task 2: `mimic/features.py` — funciones geométricas base

**Files:**
- Create: `mimic/features.py`
- Test: `tests/mimic/test_features.py`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/mimic/test_features.py`:
```python
import numpy as np
import pytest
from mimic.features import angulo, distancia

def test_angulo_90_grados():
    # vertice en el origen, un brazo hacia arriba, otro hacia la derecha
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angulo(a, b, c) == pytest.approx(90.0, abs=0.1)

def test_angulo_180_grados():
    # los tres puntos alineados: el angulo en b debe ser 180
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angulo(a, b, c) == pytest.approx(180.0, abs=0.1)

def test_distancia_pitagoras():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert distancia(a, b) == pytest.approx(5.0, abs=1e-6)
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `pytest tests/mimic/test_features.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'mimic.features'`

- [ ] **Step 3: Implementar lo mínimo para que pasen**

`mimic/features.py`:
```python
"""Funciones geometricas puras sobre puntos 2D normalizados de MediaPipe.

Se trabaja siempre con arreglos numpy de forma (2,) -- x, y en
coordenadas normalizadas de la imagen (0 a 1), tal como las entrega
HolisticLandmarker. No dependen de mediapipe directamente, para poder
probarlas con puntos inventados.
"""
import numpy as np


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo en grados formado en el vertice b por los segmentos b-a y b-c."""
    ba = a - b
    bc = c - b
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    coseno = np.clip(coseno, -1.0, 1.0)
    return float(np.degrees(np.arccos(coseno)))


def distancia(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `pytest tests/mimic/test_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add mimic/features.py tests/mimic/test_features.py
git commit -m "Agregar funciones geometricas base: angulo y distancia"
```

---

## Task 3: `mimic/features.py` — vector de features de 10 dimensiones

**Files:**
- Modify: `mimic/features.py`
- Test: `tests/mimic/test_features.py`

**Contexto de índices de MediaPipe usados** (ya verificados en el smoke test local): pose de 33 puntos (`11`=hombro izq, `12`=hombro der, `13`=codo izq, `14`=codo der, `15`=muñeca izq, `16`=muñeca der, `0`=nariz), malla facial (`152`=mentón).

- [ ] **Step 1: Escribir el test que falla**

Añadir a `tests/mimic/test_features.py`:
```python
from types import SimpleNamespace
from mimic.features import construir_vector_features, NOMBRES_FEATURES

def _punto(x, y):
    return SimpleNamespace(x=x, y=y)

def test_construir_vector_features_devuelve_10_valores_nombrados():
    # Postura simetrica e inventada: brazos hacia abajo, manos separadas
    pose = {
        11: _punto(0.4, 0.3),  # hombro izq
        12: _punto(0.6, 0.3),  # hombro der
        13: _punto(0.35, 0.45),  # codo izq
        14: _punto(0.65, 0.45),  # codo der
        15: _punto(0.3, 0.6),  # muneca izq
        16: _punto(0.7, 0.6),  # muneca der
        0: _punto(0.5, 0.15),  # nariz
    }
    menton = _punto(0.5, 0.2)

    vector = construir_vector_features(pose, menton)

    assert len(vector) == 10
    assert len(NOMBRES_FEATURES) == 10
    assert all(isinstance(v, float) for v in vector)

def test_construir_vector_features_lanza_error_si_falta_un_punto_clave():
    pose_incompleta = {11: _punto(0.4, 0.3)}  # faltan los demas puntos
    with pytest.raises(KeyError):
        construir_vector_features(pose_incompleta, _punto(0.5, 0.2))
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_features.py -v`
Expected: FAIL con `ImportError: cannot import name 'construir_vector_features'`

- [ ] **Step 3: Implementar**

Añadir a `mimic/features.py`:
```python
HOMBRO_IZQ, HOMBRO_DER = 11, 12
CODO_IZQ, CODO_DER = 13, 14
MUNECA_IZQ, MUNECA_DER = 15, 16
NARIZ = 0

NOMBRES_FEATURES = [
    "angulo_codo_izq",
    "angulo_codo_der",
    "angulo_hombro_izq",
    "angulo_hombro_der",
    "inclinacion_cabeza",
    "distancia_manos",
    "distancia_mano_menton",
    "distancia_mano_hombro_contrario",
    "altura_manos_relativa",
    "asimetria_manos",
]


def _p(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y])


def construir_vector_features(pose: dict, menton) -> list[float]:
    """Convierte landmarks crudos en el vector de 10 features de MIMIC.

    'pose' es un diccionario {indice: landmark} con al menos hombros,
    codos, munecas y nariz -- lo que HolisticLandmarker entrega siempre
    que el torso superior sea visible, sin importar si es una foto de
    cuerpo completo o un plano de medio cuerpo como en HaGRID.
    """
    hombro_izq, hombro_der = _p(pose[HOMBRO_IZQ]), _p(pose[HOMBRO_DER])
    codo_izq, codo_der = _p(pose[CODO_IZQ]), _p(pose[CODO_DER])
    muneca_izq, muneca_der = _p(pose[MUNECA_IZQ]), _p(pose[MUNECA_DER])
    nariz = _p(pose[NARIZ])
    menton_p = _p(menton)

    ancho_hombros = distancia(hombro_izq, hombro_der) + 1e-9
    hombro_medio = (hombro_izq + hombro_der) / 2
    vertical = np.array([0.0, 1.0])

    angulo_codo_izq = angulo(hombro_izq, codo_izq, muneca_izq)
    angulo_codo_der = angulo(hombro_der, codo_der, muneca_der)
    angulo_hombro_izq = angulo(codo_izq, hombro_izq, hombro_izq + vertical)
    angulo_hombro_der = angulo(codo_der, hombro_der, hombro_der + vertical)
    inclinacion_cabeza = angulo(nariz, hombro_medio, hombro_medio + vertical)

    distancia_manos = distancia(muneca_izq, muneca_der) / ancho_hombros
    distancia_mano_menton = min(
        distancia(muneca_izq, menton_p), distancia(muneca_der, menton_p)
    ) / ancho_hombros
    distancia_mano_hombro_contrario = min(
        distancia(muneca_izq, hombro_der), distancia(muneca_der, hombro_izq)
    ) / ancho_hombros
    altura_manos_relativa = (
        (muneca_izq[1] + muneca_der[1]) / 2 - hombro_medio[1]
    ) / ancho_hombros
    asimetria_manos = abs(muneca_izq[1] - muneca_der[1]) / ancho_hombros

    return [
        angulo_codo_izq,
        angulo_codo_der,
        angulo_hombro_izq,
        angulo_hombro_der,
        inclinacion_cabeza,
        distancia_manos,
        distancia_mano_menton,
        distancia_mano_hombro_contrario,
        altura_manos_relativa,
        asimetria_manos,
    ]
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_features.py -v`
Expected: PASS (5 tests en total)

- [ ] **Step 5: Commit**

```bash
git add mimic/features.py tests/mimic/test_features.py
git commit -m "Agregar construccion del vector de 10 features geometricas"
```

---

## Task 4: `mimic/temporal.py` — suavizado de etiquetas por ventana

**Files:**
- Create: `mimic/temporal.py`
- Test: `tests/mimic/test_temporal.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/mimic/test_temporal.py`:
```python
from mimic.temporal import SuavizadorTemporal

def test_devuelve_la_clase_mayoritaria_en_la_ventana():
    suavizador = SuavizadorTemporal(tamano_ventana=5)
    etiquetas = ["neutral", "neutral", "zen", "neutral", "zen"]
    resultado = None
    for etiqueta in etiquetas:
        resultado = suavizador.actualizar(etiqueta)
    assert resultado == "neutral"

def test_ventana_deslizante_olvida_predicciones_viejas():
    suavizador = SuavizadorTemporal(tamano_ventana=3)
    for etiqueta in ["neutral", "neutral", "neutral"]:
        suavizador.actualizar(etiqueta)
    # las siguientes 3 predicciones deberian desplazar del todo a "neutral"
    resultado = None
    for etiqueta in ["zen", "zen", "zen"]:
        resultado = suavizador.actualizar(etiqueta)
    assert resultado == "zen"
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_temporal.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/temporal.py`:
```python
"""Suaviza predicciones frame a frame para evitar que la etiqueta
parpadee entre clases parecidas de un frame a otro."""
from collections import deque, Counter


class SuavizadorTemporal:
    def __init__(self, tamano_ventana: int = 7):
        self._ventana = deque(maxlen=tamano_ventana)

    def actualizar(self, etiqueta: str) -> str:
        self._ventana.append(etiqueta)
        conteo = Counter(self._ventana)
        return conteo.most_common(1)[0][0]
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_temporal.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mimic/temporal.py tests/mimic/test_temporal.py
git commit -m "Agregar suavizado temporal de etiquetas por ventana deslizante"
```

---

## Task 5: `mimic/landmarks.py` — envoltorio de HolisticLandmarker

**Files:**
- Create: `mimic/landmarks.py`
- Test: `tests/mimic/test_landmarks.py`

- [ ] **Step 1: Escribir el test que falla (usa la fixture real del Task 1)**

`tests/mimic/test_landmarks.py`:
```python
from pathlib import Path
import cv2
from mimic.landmarks import DetectorHolistic

FIXTURE = Path(__file__).parent.parent / "fixtures" / "persona_prueba.jpg"

def test_detecta_landmarks_de_pose_y_rostro_en_foto_real():
    detector = DetectorHolistic()
    frame = cv2.imread(str(FIXTURE))
    resultado = detector.procesar(frame)

    assert resultado.pose is not None
    assert 11 in resultado.pose  # hombro izquierdo detectado
    assert resultado.menton is not None
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_landmarks.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/landmarks.py`:
```python
"""Envoltorio delgado sobre mediapipe.tasks.python.vision.HolisticLandmarker.

Nota importante descubierta durante el desarrollo: la API vigente de
mediapipe (0.10.35) ya no expone mp.solutions.holistic -- ese modulo
antiguo fue reemplazado por la API de Tasks. Aqui se usa
HolisticLandmarker, que requiere descargar un archivo .task de modelo
(se hace una sola vez, no se sube al repositorio).
"""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
RUTA_MODELO = Path("models/holistic_landmarker.task")

MENTON_INDICE = 152  # indice del menton en la malla facial de mediapipe


@dataclass
class ResultadoLandmarks:
    pose: dict | None  # {indice: landmark} o None si no se detecto a nadie
    menton: object | None


def _asegurar_modelo_descargado() -> Path:
    if not RUTA_MODELO.exists():
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL_MODELO, RUTA_MODELO)
    return RUTA_MODELO


class DetectorHolistic:
    def __init__(self):
        ruta_modelo = _asegurar_modelo_descargado()
        opciones = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ruta_modelo)),
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opciones)

    def procesar(self, frame_bgr) -> ResultadoLandmarks:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = self._landmarker.detect(imagen_mp)

        pose = None
        if resultado.pose_landmarks:
            pose = {i: lm for i, lm in enumerate(resultado.pose_landmarks[0])}

        menton = None
        if resultado.face_landmarks:
            menton = resultado.face_landmarks[0][MENTON_INDICE]

        return ResultadoLandmarks(pose=pose, menton=menton)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_landmarks.py -v`
Expected: PASS (1 test). La primera corrida descarga el modelo (~14 MB); las siguientes lo reusan desde `models/`.

- [ ] **Step 5: Commit**

```bash
git add mimic/landmarks.py tests/mimic/test_landmarks.py
git commit -m "Agregar envoltorio de HolisticLandmarker para extraer pose, manos y menton"
```

---

## Task 6: `mimic/classifier.py` — entrenamiento y selección de modelo

**Files:**
- Create: `mimic/classifier.py`
- Test: `tests/mimic/test_classifier.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/mimic/test_classifier.py`:
```python
import numpy as np
from mimic.classifier import entrenar_y_seleccionar

def test_entrena_y_selecciona_el_mejor_modelo_por_f1_macro():
    # dos clases perfectamente separables en el espacio de features
    rng = np.random.default_rng(42)
    clase_a = rng.normal(loc=0.0, scale=0.1, size=(30, 10))
    clase_b = rng.normal(loc=5.0, scale=0.1, size=(30, 10))
    X = np.vstack([clase_a, clase_b])
    y = ["neutral"] * 30 + ["zen"] * 30

    resultado = entrenar_y_seleccionar(X, y)

    assert resultado.modelo is not None
    assert resultado.nombre_modelo in ("svm", "random_forest")
    assert resultado.f1_macro_validacion > 0.9

def test_guarda_y_carga_el_modelo(tmp_path):
    from mimic.classifier import guardar_modelo, cargar_modelo

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 10)), rng.normal(5, 0.1, (20, 10))])
    y = ["a"] * 20 + ["b"] * 20
    resultado = entrenar_y_seleccionar(X, y)

    ruta = tmp_path / "modelo.joblib"
    guardar_modelo(resultado.modelo, ruta)
    modelo_cargado = cargar_modelo(ruta)

    prediccion = modelo_cargado.predict(X[:1])
    assert prediccion[0] in ("a", "b")
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_classifier.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/classifier.py`:
```python
"""Entrena y compara un SVM y un Random Forest sobre el vector de
features de MIMIC, y se queda con el que tenga mejor F1 macro --no
accuracy global-- porque las clases estan desbalanceadas (HaGRID aporta
muchas mas muestras que las clases curadas a mano."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC


@dataclass
class ResultadoEntrenamiento:
    modelo: object
    nombre_modelo: str
    f1_macro_validacion: float


def entrenar_y_seleccionar(X, y) -> ResultadoEntrenamiento:
    candidatos = {
        "svm": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    n_clases_minimo = min(np.unique(y, return_counts=True)[1])
    n_splits = max(2, min(5, n_clases_minimo))
    validador = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mejor_nombre, mejor_modelo, mejor_f1 = None, None, -1.0
    for nombre, modelo in candidatos.items():
        puntajes = cross_val_score(modelo, X, y, cv=validador, scoring="f1_macro")
        f1_promedio = float(np.mean(puntajes))
        if f1_promedio > mejor_f1:
            mejor_nombre, mejor_modelo, mejor_f1 = nombre, modelo, f1_promedio

    mejor_modelo.fit(X, y)
    return ResultadoEntrenamiento(
        modelo=mejor_modelo, nombre_modelo=mejor_nombre, f1_macro_validacion=mejor_f1
    )


def guardar_modelo(modelo, ruta: Path) -> None:
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path):
    return joblib.load(ruta)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_classifier.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mimic/classifier.py tests/mimic/test_classifier.py
git commit -m "Agregar entrenamiento y seleccion de modelo (SVM vs Random Forest) por F1 macro"
```

---

## Task 7: Manifiesto de fuentes de datos (10 clases)

**Files:**
- Create: `mimic/data_curation/__init__.py`
- Create: `mimic/data_curation/manifiesto.py`
- Test: `tests/mimic/test_manifiesto.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/mimic/test_manifiesto.py`:
```python
from mimic.data_curation.manifiesto import CLASES

def test_manifiesto_tiene_10_clases():
    assert len(CLASES) == 10

def test_cada_clase_declara_su_tipo_y_fuente():
    tipos_validos = {"gesto", "pose"}
    fuentes_validas = {"hagrid", "fotos_cc"}
    for clase in CLASES:
        assert clase.tipo in tipos_validos
        assert clase.fuente in fuentes_validas

def test_las_dos_clases_de_hagrid_estan_declaradas():
    nombres_hagrid = {c.nombre for c in CLASES if c.fuente == "hagrid"}
    assert nombres_hagrid == {"saludo", "pulgar_arriba"}
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_manifiesto.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/data_curation/manifiesto.py`:
```python
"""Manifiesto de las 10 clases de MIMIC: de donde viene cada una y con
que etiqueta original se busca en su fuente. Ver docs/superpowers/specs/
2026-07-05-mimic-design.md seccion 2 para la justificacion completa."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinicionClase:
    nombre: str
    tipo: str  # "gesto" o "pose"
    fuente: str  # "hagrid" o "fotos_cc"
    etiqueta_hagrid: str | None = None  # solo si fuente == "hagrid"


CLASES = [
    DefinicionClase("saludo", "gesto", "hagrid", etiqueta_hagrid="palm"),
    DefinicionClase("pulgar_arriba", "gesto", "hagrid", etiqueta_hagrid="like"),
    DefinicionClase("mano_en_menton", "gesto", "fotos_cc"),
    DefinicionClase("manos_juntas", "gesto", "fotos_cc"),
    DefinicionClase("senalamiento", "gesto", "fotos_cc"),
    DefinicionClase("neutral", "pose", "fotos_cc"),
    DefinicionClase("zen", "pose", "fotos_cc"),
    DefinicionClase("pensando", "pose", "fotos_cc"),
    DefinicionClase("brazos_cruzados", "pose", "fotos_cc"),
    DefinicionClase("brazos_abiertos", "pose", "fotos_cc"),
]
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_manifiesto.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add mimic/data_curation/ tests/mimic/test_manifiesto.py
git commit -m "Agregar manifiesto de las 10 clases de MIMIC y sus fuentes"
```

---

## Task 8: Curación real de datos — investigación y descarga

Esta tarea es distinta a las anteriores: no es TDD sobre lógica pura, es trabajo de obtención de datos reales. Se ejecuta manualmente (no por un subagente ciego) porque requiere verificar visualmente que cada foto curada realmente corresponde a la clase.

- [ ] **Step 1: Descargar un subconjunto de HaGRID (clases `palm` y `like`) vía la librería `datasets` de Hugging Face, filtrando esas dos clases, ~120 imágenes cada una.**
- [ ] **Step 2: Buscar y descargar 20-40 fotos con licencia CC/uso libre (Pexels, Pixabay, Wikimedia Commons) para cada una de las 8 clases restantes, verificando visualmente que cada foto muestra la pose/gesto descrito antes de guardarla.**
- [ ] **Step 3: Escribir `mimic/data_curation/construir_metadata.py`, que recorre `data/images/<clase>/` y genera `data/metadata.csv` con columnas `sample_id, clase, tipo, fuente, licencia, url_origen, split`.**
- [ ] **Step 4: Ejecutar el split estratificado 70/15/15 y dejarlo grabado en la columna `split` de `metadata.csv`.**
- [ ] **Step 5: Commit del script de curación y de `data/metadata.csv` (las imágenes en sí quedan fuera de git, según `.gitignore`).**

```bash
git add mimic/data_curation/construir_metadata.py data/metadata.csv
git commit -m "Agregar script de curacion de dataset y metadata.csv con las 10 clases"
```

---

## Task 9: `mimic/pipeline.py` — integración a `PerceptionResult`

**Files:**
- Create: `mimic/pipeline.py`
- Test: `tests/mimic/test_pipeline.py`

- [ ] **Step 1: Escribir el test que falla (usa dobles/mocks del detector y el modelo, no el modelo real todavía)**

`tests/mimic/test_pipeline.py`:
```python
from unittest.mock import MagicMock
from types import SimpleNamespace
import numpy as np
from mimic.pipeline import procesar_frame

def test_procesar_frame_devuelve_perception_result_con_etiqueta():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(
        pose={
            11: SimpleNamespace(x=0.4, y=0.3), 12: SimpleNamespace(x=0.6, y=0.3),
            13: SimpleNamespace(x=0.35, y=0.45), 14: SimpleNamespace(x=0.65, y=0.45),
            15: SimpleNamespace(x=0.3, y=0.6), 16: SimpleNamespace(x=0.7, y=0.6),
            0: SimpleNamespace(x=0.5, y=0.15),
        },
        menton=SimpleNamespace(x=0.5, y=0.2),
    )
    modelo_falso = MagicMock()
    modelo_falso.predict.return_value = ["neutral"]
    modelo_falso.predict_proba.return_value = np.array([[0.9, 0.1]])
    modelo_falso.classes_ = ["neutral", "zen"]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resultado = procesar_frame(frame, detector_falso, modelo_falso, timestamp=1.5)

    assert resultado["etiqueta_pose"] == "neutral"
    assert resultado["confianza"] == 0.9
    assert resultado["timestamp"] == 1.5
    assert resultado["frame"] is frame

def test_procesar_frame_sin_persona_devuelve_etiqueta_none():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(pose=None, menton=None)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resultado = procesar_frame(frame, detector_falso, modelo=MagicMock(), timestamp=0.0)

    assert resultado["etiqueta_pose"] is None
    assert resultado["confianza"] == 0.0
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_pipeline.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/pipeline.py`:
```python
"""Une deteccion de landmarks, features y clasificacion en un solo
resultado por frame (PerceptionResult), tal como lo definimos en el
diseno transversal del proyecto."""
import numpy as np

from mimic.features import construir_vector_features


def procesar_frame(frame, detector, modelo, timestamp: float) -> dict:
    resultado_landmarks = detector.procesar(frame)

    if resultado_landmarks.pose is None:
        return {
            "frame": frame,
            "bbox_persona": None,
            "landmarks_normalizados": None,
            "etiqueta_pose": None,
            "confianza": 0.0,
            "timestamp": timestamp,
        }

    vector = construir_vector_features(resultado_landmarks.pose, resultado_landmarks.menton)
    vector = np.array(vector).reshape(1, -1)

    etiqueta = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    indice_clase = list(modelo.classes_).index(etiqueta)
    confianza = float(probabilidades[indice_clase])

    return {
        "frame": frame,
        "bbox_persona": None,
        "landmarks_normalizados": vector.tolist()[0],
        "etiqueta_pose": etiqueta,
        "confianza": confianza,
        "timestamp": timestamp,
    }
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add mimic/pipeline.py tests/mimic/test_pipeline.py
git commit -m "Agregar pipeline que integra landmarks, features y clasificacion en PerceptionResult"
```

---

## Task 10: `mimic/capture.py` — fuente de frames

**Files:**
- Create: `mimic/capture.py`
- Test: `tests/mimic/test_capture.py`

- [ ] **Step 1: Escribir el test que falla (sobre un video de prueba generado en el momento, no sobre la webcam real)**

`tests/mimic/test_capture.py`:
```python
from pathlib import Path
import cv2
import numpy as np
from mimic.capture import leer_frames_de_video

def _crear_video_de_prueba(ruta: Path, num_frames: int = 5):
    escritor = cv2.VideoWriter(
        str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
    )
    for _ in range(num_frames):
        escritor.write(np.zeros((64, 64, 3), dtype=np.uint8))
    escritor.release()

def test_lee_todos_los_frames_de_un_video(tmp_path):
    ruta_video = tmp_path / "prueba.mp4"
    _crear_video_de_prueba(ruta_video, num_frames=5)

    frames = list(leer_frames_de_video(str(ruta_video)))

    assert len(frames) == 5
    assert frames[0].shape == (64, 64, 3)
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/mimic/test_capture.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`mimic/capture.py`:
```python
"""Fuentes de frames para MIMIC: archivo de video o webcam local. La
demo Gradio no pasa por aqui -- usa su propio componente de camara del
navegador (ver app/app.py)."""
import cv2


def leer_frames_de_video(ruta: str):
    captura = cv2.VideoCapture(ruta)
    try:
        while True:
            hay_frame, frame = captura.read()
            if not hay_frame:
                break
            yield frame
    finally:
        captura.release()


def leer_frames_de_webcam(indice_camara: int = 0):
    captura = cv2.VideoCapture(indice_camara)
    try:
        while True:
            hay_frame, frame = captura.read()
            if not hay_frame:
                break
            yield frame
    finally:
        captura.release()
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/mimic/test_capture.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add mimic/capture.py tests/mimic/test_capture.py
git commit -m "Agregar lectura de frames desde video o webcam"
```

---

## Task 11: `mimic/live_demo.py` — script OpenCV en vivo con FPS

**Files:**
- Create: `mimic/live_demo.py`

No lleva TDD formal (depende de una camara fisica y de una ventana de video, no es automatizable en CI). Se verifica manualmente ejecutándolo en local.

- [ ] **Step 1: Implementar**

`mimic/live_demo.py`:
```python
"""Demo de escritorio con OpenCV puro, para medir FPS reales sin la
latencia de red que agrega Gradio. Se ejecuta solo en local:
    python -m mimic.live_demo
"""
import time

import cv2

from mimic.capture import leer_frames_de_webcam
from mimic.classifier import cargar_modelo
from mimic.landmarks import DetectorHolistic
from mimic.pipeline import procesar_frame
from mimic.temporal import SuavizadorTemporal

RUTA_MODELO = "models/mimic_clasificador.joblib"


def main():
    detector = DetectorHolistic()
    modelo = cargar_modelo(RUTA_MODELO)
    suavizador = SuavizadorTemporal()

    tiempo_anterior = time.time()
    for frame in leer_frames_de_webcam():
        resultado = procesar_frame(frame, detector, modelo, timestamp=time.time())

        etiqueta_estable = "..."
        if resultado["etiqueta_pose"] is not None:
            etiqueta_estable = suavizador.actualizar(resultado["etiqueta_pose"])

        ahora = time.time()
        fps = 1.0 / max(ahora - tiempo_anterior, 1e-6)
        tiempo_anterior = ahora

        texto = f"{etiqueta_estable} ({resultado['confianza']:.2f}) - {fps:.1f} FPS"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("MIMIC - live demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add mimic/live_demo.py
git commit -m "Agregar demo de escritorio con OpenCV para medir FPS reales en vivo"
```

---

## Task 12: Notebook `D1_mimic_baseline.ipynb`

- [ ] **Step 1:** Crear el notebook con celdas: detección de entorno (Colab/local), instalación de `requirements.txt` si es Colab, carga de `data/metadata.csv`, extracción de landmarks y features sobre todo el dataset curado, entrenamiento con `entrenar_y_seleccionar`, evaluación (matriz de confusión, precision/recall/F1 por clase vía `sklearn.metrics.classification_report`), guardado del modelo en `models/mimic_clasificador.joblib`, y una celda de demo con `gr.Image(sources=["webcam"])` llamando a `procesar_frame`.
- [ ] **Step 2:** Ejecutar el notebook de punta a punta sobre el dataset real curado en el Task 8 y confirmar que corre sin errores.
- [ ] **Step 3:** Commit

```bash
git add notebooks/D1_mimic_baseline.ipynb
git commit -m "Agregar notebook D1_mimic_baseline con pipeline completo y metricas"
```

---

## Task 13: `app/app.py` — pestaña Gradio de MIMIC

**Files:**
- Create: `app/app.py`

- [ ] **Step 1: Implementar la pestaña MIMIC**

```python
"""Interfaz Gradio de MimicVision AI. Por ahora solo tiene la pestana
de MIMIC -- MATCH y ASK se agregan en sus propios sub-proyectos sin
tocar esta pestana."""
import gradio as gr

from mimic.classifier import cargar_modelo
from mimic.landmarks import DetectorHolistic
from mimic.pipeline import procesar_frame
from mimic.temporal import SuavizadorTemporal

_detector = DetectorHolistic()
_modelo = cargar_modelo("models/mimic_clasificador.joblib")
_suavizador = SuavizadorTemporal()


def _clasificar_frame(imagen):
    if imagen is None:
        return "Esperando imagen..."
    resultado = procesar_frame(imagen, _detector, _modelo, timestamp=0.0)
    if resultado["etiqueta_pose"] is None:
        return "No se detecto a nadie en la imagen"
    etiqueta_estable = _suavizador.actualizar(resultado["etiqueta_pose"])
    return f"{etiqueta_estable} (confianza: {resultado['confianza']:.2f})"


with gr.Blocks(title="MimicVision AI") as demo:
    with gr.Tab("MIMIC"):
        entrada = gr.Image(sources=["webcam"], streaming=True)
        salida = gr.Textbox(label="Pose o gesto detectado")
        entrada.stream(_clasificar_frame, inputs=entrada, outputs=salida)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Verificar manualmente que `python app/app.py` levanta la interfaz sin errores de import (la cámara en sí se prueba a mano, no en CI).**

Run: `python -c "import app.app"`
Expected: no lanza excepciones al importar (confirma que todos los módulos existen y encajan).

- [ ] **Step 3: Commit**

```bash
git add app/app.py
git commit -m "Agregar demo Gradio de la pestana MIMIC"
```

---

## Task 14: README

- [ ] **Step 1:** Escribir `README.md` con: descripción del proyecto, instrucciones de instalación local (`python -m venv .venv`, activar, `pip install -r requirements.txt`), instrucciones para Colab (abrir el notebook, la celda de detección de entorno instala automáticamente), cómo correr los tests (`pytest`), cómo correr la demo (`python app/app.py` o `python -m mimic.live_demo`), y estructura de carpetas.
- [ ] **Step 2:** Commit

```bash
git add README.md
git commit -m "Agregar README con instrucciones de instalacion y ejecucion"
```

---

## Self-Review (completado antes de ejecutar)

**Cobertura del spec:** dataset (Task 7-8), MediaPipe Holistic (Task 5), features de 10 dimensiones (Task 3), clasificador SVM vs RF con F1 macro (Task 6), tiempo real + FPS (Task 10-11), notebook (Task 12), demo Gradio (Task 13), README (Task 14). Todo el spec de MIMIC queda cubierto.

**Placeholders:** ninguno — cada paso tiene código completo, no hay "TBD" ni "implementar después".

**Consistencia de tipos:** `PerceptionResult` como `dict` con las mismas claves en `pipeline.py`, se usa igual en `live_demo.py` y `app.py`. `NOMBRES_FEATURES` y el orden del vector en `construir_vector_features` coinciden. `DetectorHolistic.procesar()` devuelve siempre `ResultadoLandmarks` con los mismos campos (`pose`, `menton`), consumido igual en `pipeline.py` y en el test.
