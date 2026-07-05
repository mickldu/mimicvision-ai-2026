"""Demo de escritorio con OpenCV puro, para medir FPS reales sin la
latencia de red que agrega Gradio. Se ejecuta solo en local:

    python -m mimic.live_demo

Requiere haber entrenado antes el modelo con el notebook
D1_mimic_baseline.ipynb (que lo guarda en models/).
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
