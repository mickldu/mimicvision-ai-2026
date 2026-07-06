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
