"""Interfaz Gradio de MimicVision AI. Tiene las pestanas de MIMIC y
MATCH -- ASK se agregara en su propio sub-proyecto sin tocar estas dos.

Funciona igual en local y en Colab: la camara se captura con el
componente de webcam de Gradio, que usa el navegador y no depende de
cv2.VideoCapture. En local tambien existe mimic/live_demo.py para
medir FPS reales sin la latencia del navegador.

Uso:
    python -m app.app
"""
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from mimic.classifier import cargar_modelo
from mimic.landmarks import DetectorHolistic
from mimic.pipeline import procesar_frame
from mimic.temporal import SuavizadorTemporal

RUTA_MODELO = Path("models/mimic_clasificador.joblib")

_detector = None
_modelo = None
_suavizador = SuavizadorTemporal()
_indice_match = None


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


def _inicializar_match():
    """Carga perezosa del indice de MATCH: construye la galeria de
    embeddings SigLIP2 la primera vez que alguien busca, no al importar
    el modulo (mismo patron que _inicializar() de MIMIC)."""
    global _indice_match
    if _indice_match is not None:
        return

    from match.embeddings import ExtractorSigLIP2
    from match.gallery import cargar_galeria_y_consultas
    from match.index import IndiceSimilitud

    galeria, _ = cargar_galeria_y_consultas()
    extractor = ExtractorSigLIP2()
    vectores, metadatos = [], []
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


def buscar_similares(imagen):
    if imagen is None:
        return []
    _inicializar_match()
    indice, extractor = _indice_match
    vector_consulta = extractor.extraer(imagen[:, :, ::-1])
    resultados = indice.buscar(vector_consulta, k=5)
    return [r["ruta_archivo"] for r in resultados]


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
        with gr.Tab("MATCH"):
            gr.Markdown(
                "Sube o captura una foto y el sistema busca las 5 imagenes "
                "mas parecidas en la galeria de MIMIC."
            )
            entrada_match = gr.Image(sources=["upload", "webcam"])
            galeria_resultados = gr.Gallery(label="Top 5 mas similares", columns=5)
            boton_buscar = gr.Button("Buscar")
            boton_buscar.click(buscar_similares, inputs=entrada_match, outputs=galeria_resultados)
    return demo


if __name__ == "__main__":
    construir_demo().launch()
