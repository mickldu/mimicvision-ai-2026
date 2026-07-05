"""Interfaz Gradio de MimicVision AI. Por ahora solo tiene la pestana
de MIMIC -- MATCH y ASK se agregaran en sus propios sub-proyectos sin
tocar esta pestana.

Funciona igual en local y en Colab: la camara se captura con el
componente de webcam de Gradio, que usa el navegador y no depende de
cv2.VideoCapture. En local tambien existe mimic/live_demo.py para
medir FPS reales sin la latencia del navegador.

Uso:
    python -m app.app
"""
from pathlib import Path

import gradio as gr

from mimic.classifier import cargar_modelo
from mimic.landmarks import DetectorHolistic
from mimic.pipeline import procesar_frame
from mimic.temporal import SuavizadorTemporal

RUTA_MODELO = Path("models/mimic_clasificador.joblib")

_detector = None
_modelo = None
_suavizador = SuavizadorTemporal()


def _inicializar():
    """Carga perezosa: el detector y el modelo se cargan en el primer
    frame y no al importar, para que el modulo se pueda importar (y
    testear) sin haber entrenado todavia."""
    global _detector, _modelo
    if _detector is None:
        _detector = DetectorHolistic()
    if _modelo is None:
        if not RUTA_MODELO.exists():
            raise FileNotFoundError(
                f"No existe {RUTA_MODELO}. Entrena primero con el notebook "
                "notebooks/D1_mimic_baseline.ipynb"
            )
        _modelo = cargar_modelo(RUTA_MODELO)


def clasificar_frame(imagen):
    if imagen is None:
        return "Esperando imagen..."
    _inicializar()
    # Gradio entrega RGB pero el pipeline espera BGR como OpenCV
    resultado = procesar_frame(imagen[:, :, ::-1], _detector, _modelo, timestamp=0.0)
    if resultado["etiqueta_pose"] is None:
        return "No se detecto a nadie en la imagen"
    etiqueta_estable = _suavizador.actualizar(resultado["etiqueta_pose"])
    return f"{etiqueta_estable} (confianza: {resultado['confianza']:.2f})"


def construir_demo() -> gr.Blocks:
    with gr.Blocks(title="MimicVision AI") as demo:
        gr.Markdown("# MimicVision AI")
        with gr.Tab("MIMIC"):
            gr.Markdown(
                "Clasificacion de poses y gestos en tiempo real. "
                "Activa la camara y manten la pose un par de segundos."
            )
            entrada = gr.Image(sources=["webcam"], streaming=True)
            salida = gr.Textbox(label="Pose o gesto detectado")
            entrada.stream(clasificar_frame, inputs=entrada, outputs=salida)
    return demo


if __name__ == "__main__":
    construir_demo().launch()
