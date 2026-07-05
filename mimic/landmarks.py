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
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
RUTA_MODELO = Path("models/holistic_landmarker.task")

MENTON_INDICE = 152  # indice del menton en la malla facial de mediapipe

LADO_LIENZO = 640  # tamano fijo del lienzo interno (ver procesar)


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
        # El landmarker de mediapipe 0.10.35 arrastra estado interno de
        # segmentacion entre llamadas y crashea si dos frames seguidos
        # tienen dimensiones distintas. Para blindarlo, cada frame se
        # coloca sobre un lienzo cuadrado fijo con escala uniforme
        # (letterbox): al ser una transformacion de similitud, los
        # angulos y las distancias relativas que usan las features no
        # se distorsionan.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        alto, ancho = frame_rgb.shape[:2]
        escala = LADO_LIENZO / max(alto, ancho)
        redimensionado = cv2.resize(frame_rgb, (int(ancho * escala), int(alto * escala)))
        lienzo = np.zeros((LADO_LIENZO, LADO_LIENZO, 3), dtype=np.uint8)
        lienzo[: redimensionado.shape[0], : redimensionado.shape[1]] = redimensionado

        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=lienzo)
        resultado = self._landmarker.detect(imagen_mp)

        # HolisticLandmarker devuelve pose_landmarks y face_landmarks como
        # listas planas de puntos de una sola persona, no una lista por
        # persona detectada (a diferencia de la vieja API mp.solutions).
        pose = None
        if resultado.pose_landmarks:
            pose = dict(enumerate(resultado.pose_landmarks))

        menton = None
        if resultado.face_landmarks:
            menton = resultado.face_landmarks[MENTON_INDICE]

        return ResultadoLandmarks(pose=pose, menton=menton)
