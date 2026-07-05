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
