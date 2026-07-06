# ASK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ASK (Clase 3): un video de prueba compuesto (con verdad de terreno conocida), un timeline de eventos de MIMIC, un cliente de LocateAnything-3B para grounding por lenguaje natural, un almacén de eventos, y un motor de consulta que combina ambos — con modo batch (archivo) y modo en vivo (webcam).

**Architecture:** `ask/ingest.py` produce frames con timestamp desde un archivo o una fuente en vivo. `ask/timeline.py` corre `mimic.pipeline` sobre esos frames y consolida etiquetas estables en eventos preliminares. `ask/grounding.py` envuelve `nvidia/LocateAnything-3B` (solo corre con GPU real — no se pudo probar en este entorno; su función de parseo de cajas sí se prueba localmente). `ask/query_engine.py` recibe el cliente de grounding por inyección de dependencia y busca coincidencias sobre los keyframes de los eventos. `ask/events.py` y `ask/store.py` arman y persisten el contrato `Event`.

**Tech Stack:** Python 3.13, mediapipe/scikit-learn (ya en el proyecto), transformers + torch CUDA (LocateAnything-3B, solo Colab), pytest. Video de prueba: 5 clips con licencia Pexels concatenados con letterbox — técnica de redimensionado validada localmente antes de escribir este plan (ver spec, sección 3).

---

## Task 1: Video de prueba compuesto

**Files:**
- Create: `ask/__init__.py`
- Create: `ask/data_curation/__init__.py`
- Create: `ask/data_curation/construir_video_prueba.py`
- Test: `tests/ask/__init__.py`
- Test: `tests/ask/test_construir_video_prueba.py`

Esta tarea mezcla una función pura testeable (`ajustar_a_canvas`) con un paso de
descarga/concatenación que se ejecuta manualmente (igual patrón que la curación de
datos de MIMIC — no es TDD clásico porque depende de red y de archivos de video reales).

- [ ] **Step 1: Crear el paquete**

```bash
mkdir -p ask/data_curation tests/ask
touch ask/__init__.py ask/data_curation/__init__.py tests/ask/__init__.py
```

- [ ] **Step 2: Escribir el test que falla para la función pura de letterbox**

`tests/ask/test_construir_video_prueba.py`:
```python
import numpy as np

from ask.data_curation.construir_video_prueba import ajustar_a_canvas


def test_ajustar_a_canvas_produce_el_tamano_exacto_pedido():
    frame_vertical = np.zeros((1920, 1080, 3), dtype=np.uint8)
    resultado = ajustar_a_canvas(frame_vertical, ancho=640, alto=360)
    assert resultado.shape == (360, 640, 3)


def test_ajustar_a_canvas_no_recorta_contenido_de_un_frame_vertical():
    # Un pixel blanco en el centro horizontal del frame original debe
    # seguir estando dentro del area util (no en la zona de barras negras)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    frame[960, 540] = [255, 255, 255]  # centro exacto del frame original

    resultado = ajustar_a_canvas(frame, ancho=640, alto=360)

    fila_central = resultado[:, 320]  # columna central del canvas de salida
    assert fila_central.max() == 255  # el contenido blanco sigue presente
```

- [ ] **Step 3: Correr y verificar que falla**

Run: `pytest tests/ask/test_construir_video_prueba.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 4: Implementar**

`ask/data_curation/construir_video_prueba.py`:
```python
"""Descarga 5 clips de video con licencia libre de Pexels y los
concatena en un unico video de prueba con limites de tiempo conocidos
-- la verdad de terreno del timeline sin necesidad de anotacion
manual. Cada clip se ajusta a un lienzo comun con letterbox (no con un
recorte 'cover'): un recorte cover llego a cortar la cara en un clip
vertical de prueba, asi que se prefiere perder algo de area con barras
negras antes que perder contenido del cuerpo que MediaPipe necesita.

Uso:
    python -m ask.data_curation.construir_video_prueba
"""
import csv
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

CARPETA_VIDEOS = Path("data/videos")
RUTA_VIDEO_PRUEBA = CARPETA_VIDEOS / "video_prueba.mp4"
RUTA_REGISTRO = CARPETA_VIDEOS / "video_prueba_registro.csv"

ANCHO_CANVAS, ALTO_CANVAS = 640, 360
FPS_SALIDA = 25.0

LICENCIA_PEXELS = "Pexels License (libre para uso personal y comercial)"

SEGMENTOS = [
    {
        "nombre": "charla", "clase_esperada": "neutral",
        "url": "https://videos.pexels.com/video-files/4106314/4106314-hd_1280_720_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/a-man-talking-with-hand-gestures-4106314/",
    },
    {
        "nombre": "pulgar", "clase_esperada": "pulgar_arriba",
        "url": "https://videos.pexels.com/video-files/8627026/8627026-hd_1920_1080_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/man-giving-thumbs-up-8627026/",
    },
    {
        "nombre": "cruzados", "clase_esperada": "brazos_cruzados",
        "url": "https://videos.pexels.com/video-files/7686684/7686684-hd_1080_1920_24fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/people-crossed-arms-and-posing-7686684/",
    },
    {
        "nombre": "saludo", "clase_esperada": "saludo",
        "url": "https://videos.pexels.com/video-files/4586958/4586958-uhd_3840_2160_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/a-man-greeting-and-waving-his-hand-4586958/",
    },
    {
        "nombre": "senalar", "clase_esperada": "senalamiento",
        "url": "https://videos.pexels.com/video-files/6974223/6974223-uhd_2160_3840_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/man-pointing-at-the-camera-6974223/",
    },
]


def ajustar_a_canvas(frame: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Redimensiona un frame para que quepa completo dentro de un
    lienzo fijo (letterbox), sin recortar contenido."""
    h, w = frame.shape[:2]
    escala = min(ancho / w, alto / h)
    nuevo_w, nuevo_h = int(w * escala), int(h * escala)
    redimensionado = cv2.resize(frame, (nuevo_w, nuevo_h))
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    x0 = (ancho - nuevo_w) // 2
    y0 = (alto - nuevo_h) // 2
    lienzo[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = redimensionado
    return lienzo


def _descargar(url: str, destino: Path, intentos: int = 5) -> None:
    if destino.exists():
        return
    peticion = Request(url, headers={"User-Agent": "mimicvision-curacion/1.0"})
    for intento in range(intentos):
        try:
            with urlopen(peticion, timeout=90) as respuesta, open(destino, "wb") as archivo:
                archivo.write(respuesta.read())
            return
        except Exception:
            if intento == intentos - 1:
                raise
            time.sleep(10 * (2 ** intento))


def main():
    CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(
        str(RUTA_VIDEO_PRUEBA), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_SALIDA, (ANCHO_CANVAS, ALTO_CANVAS),
    )

    filas_registro = []
    tiempo_acumulado = 0.0

    for segmento in SEGMENTOS:
        ruta_local = CARPETA_VIDEOS / f"candidato_{segmento['nombre']}.mp4"
        _descargar(segmento["url"], ruta_local)

        captura = cv2.VideoCapture(str(ruta_local))
        inicio = tiempo_acumulado
        n_frames_escritos = 0

        while True:
            hay_frame, frame = captura.read()
            if not hay_frame:
                break
            escritor.write(ajustar_a_canvas(frame, ANCHO_CANVAS, ALTO_CANVAS))
            n_frames_escritos += 1
        captura.release()

        duracion = n_frames_escritos / FPS_SALIDA
        tiempo_acumulado += duracion
        filas_registro.append({
            "segmento": segmento["nombre"],
            "clase_esperada": segmento["clase_esperada"],
            "inicio_s": round(inicio, 2),
            "fin_s": round(tiempo_acumulado, 2),
            "licencia": LICENCIA_PEXELS,
            "pagina_origen": segmento["pagina_origen"],
        })
        print(f"{segmento['nombre']}: {inicio:.1f}s - {tiempo_acumulado:.1f}s")

    escritor.release()

    with open(RUTA_REGISTRO, "w", newline="", encoding="utf-8") as archivo:
        escritor_csv = csv.DictWriter(archivo, fieldnames=list(filas_registro[0].keys()))
        escritor_csv.writeheader()
        escritor_csv.writerows(filas_registro)

    print(f"Video de prueba: {RUTA_VIDEO_PRUEBA} ({tiempo_acumulado:.1f}s totales)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Correr el test de la funcion pura y verificar que pasa**

Run: `pytest tests/ask/test_construir_video_prueba.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Ejecutar la construcción real del video (paso manual, no automatizado)**

Run: `python -m ask.data_curation.construir_video_prueba`
Expected: descarga los 5 clips (si no existen ya en `data/videos/`) y genera
`data/videos/video_prueba.mp4` (~82s) y `data/videos/video_prueba_registro.csv`.

- [ ] **Step 7: Commit**

```bash
git add ask/__init__.py ask/data_curation/ tests/ask/__init__.py tests/ask/test_construir_video_prueba.py data/videos/video_prueba_registro.csv
git commit -m "Agregar construccion del video de prueba compuesto con verdad de terreno"
```

---

## Task 2: `ask/ingest.py` — muestreo de frames (archivo y en vivo)

**Files:**
- Create: `ask/ingest.py`
- Test: `tests/ask/test_ingest.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/ask/test_ingest.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from ask.ingest import muestrear_frames_de_video, muestrear_frames_en_vivo


def _crear_video_de_prueba(ruta: Path, num_frames: int = 50, fps: float = 25.0):
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for _ in range(num_frames):
        escritor.write(np.zeros((32, 32, 3), dtype=np.uint8))
    escritor.release()


def test_muestrear_frames_de_video_reduce_la_tasa_a_lo_pedido(tmp_path):
    ruta_video = tmp_path / "prueba.mp4"
    _crear_video_de_prueba(ruta_video, num_frames=50, fps=25.0)

    muestras = list(muestrear_frames_de_video(str(ruta_video), fps_muestreo=5.0))

    # 50 frames a 25fps duran 2s; a 5fps de muestreo esperamos ~10 muestras
    assert len(muestras) == 10
    assert muestras[0][0] == pytest.approx(0.0)
    assert muestras[1][0] == pytest.approx(0.2, abs=1e-6)


def test_muestrear_frames_en_vivo_pasa_cada_frame_con_timestamp():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

    resultado = list(muestrear_frames_en_vivo(iter(frames)))

    assert len(resultado) == 3
    assert all(isinstance(marca_tiempo, float) for marca_tiempo, _ in resultado)
    assert resultado[0][1] is frames[0]
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_ingest.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/ingest.py`:
```python
"""Ingesta de video para ASK: convierte una fuente de frames (archivo o
en vivo) en frames muestreados con timestamp, a una tasa reducida para
que el pipeline ligero de MIMIC no procese cada frame del video
original. La ruta VLM (LocateAnything) nunca ve estos frames
directamente -- solo los keyframes que arma ask/timeline.py."""
import time

import cv2


def muestrear_frames_de_video(ruta_video: str, fps_muestreo: float = 10.0):
    captura = cv2.VideoCapture(ruta_video)
    fps_origen = captura.get(cv2.CAP_PROP_FPS) or 25.0
    paso = max(1, round(fps_origen / fps_muestreo))

    indice = 0
    try:
        while True:
            hay_frame, frame = captura.read()
            if not hay_frame:
                break
            if indice % paso == 0:
                yield indice / fps_origen, frame
            indice += 1
    finally:
        captura.release()


def muestrear_frames_en_vivo(generador_frames):
    """Para una fuente en vivo (ej. webcam) no hace falta submuestrear
    por fps: la latencia del propio pipeline (landmarks + clasificacion)
    ya limita la tasa efectiva al rango ligero, y el timestamp es
    simplemente el reloj de pared en el momento de capturar el frame."""
    for frame in generador_frames:
        yield time.time(), frame
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ask/ingest.py tests/ask/test_ingest.py
git commit -m "Agregar muestreo de frames de ASK para video de archivo y fuente en vivo"
```

---

## Task 3: `ask/timeline.py` — consolidación de eventos preliminares

**Files:**
- Create: `ask/timeline.py`
- Test: `tests/ask/test_timeline.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/ask/test_timeline.py`:
```python
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ask.timeline import construir_timeline


def _pose_valida():
    return {
        11: SimpleNamespace(x=0.4, y=0.3), 12: SimpleNamespace(x=0.6, y=0.3),
        13: SimpleNamespace(x=0.35, y=0.45), 14: SimpleNamespace(x=0.65, y=0.45),
        15: SimpleNamespace(x=0.3, y=0.6), 16: SimpleNamespace(x=0.7, y=0.6),
        0: SimpleNamespace(x=0.5, y=0.15),
    }


def test_construir_timeline_agrupa_frames_consecutivos_de_la_misma_clase():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(
        pose=_pose_valida(), menton=SimpleNamespace(x=0.5, y=0.2)
    )

    modelo_falso = MagicMock()
    modelo_falso.classes_ = ["neutral", "zen"]
    secuencia = ["neutral", "neutral", "neutral", "zen", "zen", "zen"]
    modelo_falso.predict.side_effect = [[e] for e in secuencia]
    modelo_falso.predict_proba.side_effect = [
        np.array([[0.9, 0.1]]) if e == "neutral" else np.array([[0.1, 0.9]]) for e in secuencia
    ]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames_muestreados = [(i * 0.1, frame) for i in range(6)]

    eventos = construir_timeline(frames_muestreados, detector_falso, modelo_falso, tamano_ventana=1)

    assert len(eventos) == 2
    assert eventos[0]["type"] == "neutral"
    assert eventos[0]["start_time"] == pytest.approx(0.0)
    assert eventos[0]["end_time"] == pytest.approx(0.3)
    assert eventos[1]["type"] == "zen"
    assert eventos[1]["start_time"] == pytest.approx(0.3)
    assert "frame_representativo" in eventos[0]
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_timeline.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/timeline.py`:
```python
"""Construye una linea temporal de eventos preliminares aplicando el
pipeline de MIMIC sobre frames muestreados: cuando la etiqueta estable
cambia, se cierra el evento anterior y se abre uno nuevo. Esta es la
'ruta ligera' del diseno de dos velocidades -- LocateAnything no
interviene aqui, solo se usa mas adelante sobre los keyframes que este
modulo produce."""
from mimic.pipeline import procesar_frame
from mimic.temporal import SuavizadorTemporal


def construir_timeline(frames_muestreados, detector, modelo, tamano_ventana: int = 5) -> list[dict]:
    suavizador = SuavizadorTemporal(tamano_ventana=tamano_ventana)
    eventos = []
    etiqueta_actual = None
    inicio_actual = None
    frames_actual = []

    def cerrar_evento(fin):
        frame_representativo = frames_actual[len(frames_actual) // 2]
        eventos.append({
            "type": etiqueta_actual,
            "start_time": inicio_actual,
            "end_time": fin,
            "duration_s": fin - inicio_actual,
            "frame_representativo": frame_representativo,
        })

    timestamp = inicio_actual
    for timestamp, frame in frames_muestreados:
        resultado = procesar_frame(frame, detector, modelo, timestamp=timestamp)
        if resultado["etiqueta_pose"] is None:
            continue
        etiqueta_estable = suavizador.actualizar(resultado["etiqueta_pose"])

        if etiqueta_actual is None:
            etiqueta_actual, inicio_actual, frames_actual = etiqueta_estable, timestamp, [frame]
        elif etiqueta_estable != etiqueta_actual:
            cerrar_evento(timestamp)
            etiqueta_actual, inicio_actual, frames_actual = etiqueta_estable, timestamp, [frame]
        else:
            frames_actual.append(frame)

    if etiqueta_actual is not None:
        cerrar_evento(timestamp)

    return eventos
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_timeline.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add ask/timeline.py tests/ask/test_timeline.py
git commit -m "Agregar consolidacion de eventos preliminares (ruta ligera de MIMIC)"
```

---

## Task 4: `ask/grounding.py` — cliente LocateAnything-3B

**Files:**
- Create: `ask/grounding.py`
- Test: `tests/ask/test_grounding.py`

**Nota importante:** este modulo no se puede probar de punta a punta en este entorno
(requiere GPU Ampere+/CUDA). Se prueban las dos partes que sí son verificables sin
GPU: el parseo de coordenadas (función pura) y que el cliente falle con un error claro
cuando no hay GPU disponible.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/ask/test_grounding.py`:
```python
import pytest

from ask.grounding import ClienteLocateAnything, ErrorHardwareNoCompatible, parsear_cajas


def test_parsear_cajas_con_formato_separado_por_comas():
    texto = "Encontrado: <box>100,200,300,400</box>"
    assert parsear_cajas(texto, ancho=1000, alto=1000) == [(100, 200, 300, 400)]


def test_parsear_cajas_con_tags_individuales_por_coordenada():
    # La documentacion del modelo no deja 100% claro el separador exacto
    # entre coordenadas -- este test cubre el formato alternativo donde
    # cada numero va en su propia sub-etiqueta.
    texto = "<box><x1>500</x1><y1>0</y1><x2>1000</x2><y2>500</y2></box>"
    assert parsear_cajas(texto, ancho=200, alto=100) == [(100, 0, 200, 50)]


def test_parsear_cajas_ignora_texto_sin_etiqueta_box():
    assert parsear_cajas("No se encontro ninguna coincidencia.", ancho=100, alto=100) == []


def test_parsear_cajas_con_multiples_cajas():
    texto = "<box>0,0,500,500</box> y tambien <box>500,500,1000,1000</box>"
    assert parsear_cajas(texto, ancho=100, alto=100) == [(0, 0, 50, 50), (50, 50, 100, 100)]


def test_cliente_lanza_error_claro_si_no_hay_gpu_compatible():
    with pytest.raises(ErrorHardwareNoCompatible):
        ClienteLocateAnything()
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_grounding.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/grounding.py`:
```python
"""Cliente de LocateAnything-3B para grounding por lenguaje natural.

ADVERTENCIA DE VERIFICACION: este modulo no se pudo probar de punta a
punta en este entorno porque el modelo exige GPU con arquitectura
Ampere o superior (A100/H100) y Linux -- ver
docs/superpowers/specs/2026-07-05-ask-design.md seccion 2. El codigo de
carga e inferencia sigue el uso documentado en la model card de
nvidia/LocateAnything-3B en Hugging Face; la validacion real de esta
llamada queda pendiente para cuando se ejecute en Colab con GPU.

Las coordenadas de salida son enteros normalizados en [0, 1000]
envueltos en <box>...</box>. La documentacion no deja completamente
claro si los cuatro numeros van separados solo por comas o en
sub-etiquetas individuales, asi que parsear_cajas extrae todos los
enteros dentro de la etiqueta en vez de asumir un separador exacto --
mas robusto ante variaciones de formato que no se pudieron confirmar
sin acceso al modelo real.
"""
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODELO_LOCATEANYTHING = "nvidia/LocateAnything-3B"


class ErrorHardwareNoCompatible(RuntimeError):
    """El hardware disponible no puede correr LocateAnything-3B."""


def parsear_cajas(texto: str, ancho: int, alto: int) -> list[tuple[int, int, int, int]]:
    cajas = []
    for bloque in re.findall(r"<box>(.*?)</box>", texto, re.DOTALL):
        numeros = [int(n) for n in re.findall(r"-?\d+", bloque)]
        if len(numeros) < 4:
            continue
        x1, y1, x2, y2 = numeros[:4]
        cajas.append((
            round(x1 / 1000 * ancho), round(y1 / 1000 * alto),
            round(x2 / 1000 * ancho), round(y2 / 1000 * alto),
        ))
    return cajas


class ClienteLocateAnything:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ErrorHardwareNoCompatible(
                "LocateAnything-3B requiere GPU CUDA (Ampere o superior). "
                "No se detecto GPU en este entorno -- correr en Colab con GPU."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODELO_LOCATEANYTHING, trust_remote_code=True
        )
        self._procesador = AutoProcessor.from_pretrained(
            MODELO_LOCATEANYTHING, trust_remote_code=True
        )
        self._modelo = AutoModel.from_pretrained(
            MODELO_LOCATEANYTHING, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda").eval()

    def localizar(self, imagen_bgr, consulta: str) -> dict:
        imagen_rgb = imagen_bgr[:, :, ::-1]
        imagen_pil = Image.fromarray(imagen_rgb)
        prompt = f"Locate all instances matching: {consulta}."

        mensajes = [{"role": "user", "content": [
            {"type": "image", "image": imagen_pil},
            {"type": "text", "text": prompt},
        ]}]
        texto = self._procesador.py_apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        imagenes, videos = self._procesador.process_vision_info(mensajes)
        entradas = self._procesador(
            text=[texto], images=imagenes, videos=videos, return_tensors="pt"
        ).to("cuda")

        respuesta = self._modelo.generate(
            pixel_values=entradas["pixel_values"].to(torch.bfloat16),
            input_ids=entradas["input_ids"],
            attention_mask=entradas["attention_mask"],
            image_grid_hws=entradas.get("image_grid_hws"),
            tokenizer=self._tokenizer,
            max_new_tokens=512,
            generation_mode="hybrid",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

        alto, ancho = imagen_bgr.shape[:2]
        cajas = parsear_cajas(respuesta, ancho, alto)
        return {"consulta": consulta, "cajas": cajas, "texto_crudo": respuesta}
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_grounding.py -v`
Expected: PASS (5 tests) -- incluyendo la confirmación de que, en este entorno sin GPU,
`ClienteLocateAnything()` falla con el error claro en vez de un traceback confuso.

- [ ] **Step 5: Commit**

```bash
git add ask/grounding.py tests/ask/test_grounding.py
git commit -m "Agregar cliente de LocateAnything-3B (parseo probado, inferencia solo valida en Colab)"
```

---

## Task 5: `ask/events.py` — contrato Event

**Files:**
- Create: `ask/events.py`
- Test: `tests/ask/test_events.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/ask/test_events.py`:
```python
from ask.events import construir_event


def test_construir_event_con_los_campos_del_contrato():
    evento_preliminar = {
        "type": "brazos_cruzados", "start_time": 1.0, "end_time": 2.5, "duration_s": 1.5,
    }

    evento = construir_event(evento_preliminar, consulta="arms crossed", frame_path="evidence/e1.jpg", fuente="MIMIC+LocateAnything")

    assert evento["type"] == "brazos_cruzados"
    assert evento["start_time"] == 1.0
    assert evento["end_time"] == 2.5
    assert evento["duration_s"] == 1.5
    assert evento["query"] == "arms crossed"
    assert evento["frame_path"] == "evidence/e1.jpg"
    assert evento["source"] == "MIMIC+LocateAnything"
    assert evento["event_id"].startswith("EVT_")


def test_cada_event_id_es_distinto():
    evento_preliminar = {"type": "zen", "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0}
    e1 = construir_event(evento_preliminar)
    e2 = construir_event(evento_preliminar)
    assert e1["event_id"] != e2["event_id"]
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_events.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/events.py`:
```python
"""Convierte un evento preliminar de la linea temporal (mas un
resultado opcional de grounding) en el contrato Event definido en el
diseno transversal del proyecto."""
import itertools

_contador_eventos = itertools.count(1)


def construir_event(evento_preliminar: dict, consulta: str = "", frame_path: str = "", fuente: str = "MIMIC") -> dict:
    return {
        "event_id": f"EVT_{next(_contador_eventos):04d}",
        "type": evento_preliminar["type"],
        "start_time": evento_preliminar["start_time"],
        "end_time": evento_preliminar["end_time"],
        "duration_s": evento_preliminar["duration_s"],
        "query": consulta,
        "frame_path": frame_path,
        "source": fuente,
    }
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_events.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ask/events.py tests/ask/test_events.py
git commit -m "Agregar construccion del contrato Event"
```

---

## Task 6: `ask/store.py` — persistencia

**Files:**
- Create: `ask/store.py`
- Test: `tests/ask/test_store.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/ask/test_store.py`:
```python
from ask.store import cargar_eventos, guardar_eventos


def test_guardar_y_cargar_eventos_hace_round_trip(tmp_path):
    ruta = tmp_path / "event_store.json"
    eventos = [{"event_id": "EVT_0001", "type": "saludo", "start_time": 0.0}]

    guardar_eventos(eventos, str(ruta))
    cargados = cargar_eventos(str(ruta))

    assert cargados == eventos
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_store.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/store.py`:
```python
"""Persistencia de eventos en un archivo JSON simple -- un video o una
sesion en vivo generan decenas de eventos, no cientos de miles, asi que
SQLite no aporta nada a esta escala (mismo criterio de simplicidad que
llevo a usar similitud coseno en vez de FAISS en MATCH)."""
import json
from pathlib import Path


def guardar_eventos(eventos: list[dict], ruta: str = "data/event_store.json") -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(eventos, archivo, indent=2, ensure_ascii=False)


def cargar_eventos(ruta: str = "data/event_store.json") -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_store.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add ask/store.py tests/ask/test_store.py
git commit -m "Agregar persistencia de eventos en JSON"
```

---

## Task 7: `ask/query_engine.py` — motor de consulta

**Files:**
- Create: `ask/query_engine.py`
- Test: `tests/ask/test_query_engine.py`

- [ ] **Step 1: Escribir el test que falla**

`tests/ask/test_query_engine.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from ask.query_engine import responder_consulta


def _evento(tipo, inicio, fin, frame):
    return {"type": tipo, "start_time": inicio, "end_time": fin, "duration_s": fin - inicio, "frame_representativo": frame}


def test_responder_consulta_devuelve_el_primer_evento_con_coincidencia():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    eventos_preliminares = [
        _evento("neutral", 0.0, 1.0, frame),
        _evento("brazos_cruzados", 1.0, 2.0, frame),
    ]
    cliente_falso = MagicMock()
    cliente_falso.localizar.side_effect = [
        {"consulta": "arms crossed", "cajas": []},
        {"consulta": "arms crossed", "cajas": [(1, 2, 3, 4)]},
    ]

    resultado = responder_consulta("arms crossed", eventos_preliminares, cliente_falso)

    assert resultado is not None
    assert resultado["type"] == "brazos_cruzados"
    assert resultado["source"] == "MIMIC+LocateAnything"


def test_responder_consulta_devuelve_none_si_nada_coincide():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    eventos_preliminares = [_evento("neutral", 0.0, 1.0, frame)]
    cliente_falso = MagicMock()
    cliente_falso.localizar.return_value = {"consulta": "x", "cajas": []}

    assert responder_consulta("algo que no existe", eventos_preliminares, cliente_falso) is None
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/ask/test_query_engine.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar**

`ask/query_engine.py`:
```python
"""Motor de consulta de ASK: toma una pregunta en texto libre, ejecuta
grounding sobre los keyframes de los eventos ya detectados por la ruta
ligera de MIMIC (no sobre cada frame del video), y devuelve el primer
evento donde se encontro una coincidencia real.

El cliente de grounding se recibe por parametro (no se instancia aqui
dentro) para poder probar esta logica con un cliente simulado -- el
cliente real (ask.grounding.ClienteLocateAnything) solo se puede
instanciar donde haya GPU compatible."""
from ask.events import construir_event


def responder_consulta(consulta: str, eventos_preliminares: list[dict], cliente_grounding) -> dict | None:
    for evento_preliminar in eventos_preliminares:
        resultado = cliente_grounding.localizar(evento_preliminar["frame_representativo"], consulta)
        if resultado["cajas"]:
            return construir_event(evento_preliminar, consulta=consulta, fuente="MIMIC+LocateAnything")
    return None
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `pytest tests/ask/test_query_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ask/query_engine.py tests/ask/test_query_engine.py
git commit -m "Agregar motor de consulta de ASK con cliente de grounding inyectable"
```

---

## Task 8: Notebook `D3_ask_video.ipynb`

- [ ] **Step 1:** Crear el notebook con celdas: detección de entorno (igual patrón que
D1/D2, más una celda que verifica `torch.cuda.is_available()` antes de intentar cargar
LocateAnything-3B), construcción/descarga del video de prueba (`ask.data_curation.construir_video_prueba`
si no existe), timeline de MIMIC sobre el video con `ask.timeline.construir_timeline`,
evaluación de la consolidación temporal contra `data/videos/video_prueba_registro.csv`
(verdad de terreno conocida — esto corre 100% en local), intento de carga de
`ask.grounding.ClienteLocateAnything` envuelto en `try/except ErrorHardwareNoCompatible`
que imprime un mensaje claro y continúa si no hay GPU en vez de romper el notebook,
grounding sobre los 5 prompts mínimos (sección 9 del spec) si el cliente cargó, guardado
en `event_store.json`, y demo Gradio con selector de modo (archivo / en vivo).
- [ ] **Step 2:** Ejecutar el notebook de punta a punta con
`jupyter nbconvert --to notebook --execute --inplace`. En este entorno sin GPU, las
celdas de grounding deben imprimir el mensaje de hardware no compatible y el notebook
debe completarse igual sin error — eso confirma que el manejo de error funciona.
- [ ] **Step 3:** Commit

```bash
git add notebooks/D3_ask_video.ipynb
git commit -m "Agregar notebook D3_ask_video con timeline, grounding y motor de consulta"
```

---

## Task 9: Pestaña ASK en `app/app.py`

**Files:**
- Modify: `app/app.py`

- [ ] **Step 1: Añadir la pestaña sin tocar MIMIC ni MATCH**

Agregar dentro del mismo `gr.Blocks`, después de la pestaña `"MATCH"`, un selector de
modo (archivo subido vs. webcam en vivo), un campo de texto para la consulta, y un
textbox de resultado que muestra tipo de evento, timestamp y ruta de evidencia — o un
mensaje claro si `ClienteLocateAnything` no pudo cargar en el entorno actual (mismo
patrón de manejo de `ErrorHardwareNoCompatible` que en el notebook).

- [ ] **Step 2: Verificar que el modulo importa sin errores**

Run: `python -c "import app.app"`
Expected: no lanza excepciones al importar.

- [ ] **Step 3: Commit**

```bash
git add app/app.py
git commit -m "Agregar pestana ASK a la demo Gradio"
```

---

## Self-Review (completado antes de ejecutar)

**Cobertura del spec:** video de prueba con verdad de terreno (Task 1), ingesta
archivo+en vivo (Task 2), timeline/ruta ligera (Task 3), cliente LocateAnything con
parseo probado y guardia de hardware (Task 4), contrato Event (Task 5), persistencia
(Task 6), motor de consulta inyectable (Task 7), notebook con evaluación local +
manejo explícito de la limitación de GPU (Task 8), demo Gradio con modo en vivo
(Task 9). La limitación de no poder validar el grounding real en este entorno se
documenta en tres lugares (spec, código, notebook) en vez de ocultarse.

**Placeholders:** ninguno.

**Consistencia de tipos:** `Event` como `dict` con las mismas claves en
`ask/events.py`, `ask/query_engine.py` y el contrato transversal. `ClienteLocateAnything.localizar()`
devuelve siempre `{"consulta", "cajas", "texto_crudo"}`, consumido igual en
`query_engine.py` y en los tests con el cliente simulado.
