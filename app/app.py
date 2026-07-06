"""Interfaz Gradio de MimicVision AI. Tiene las pestanas de MIMIC,
MATCH y ASK -- cada una se agrego en su propio sub-proyecto sin tocar
las anteriores.

Funciona igual en local y en Colab: la camara se captura con el
componente de webcam de Gradio, que usa el navegador y no depende de
cv2.VideoCapture. En local tambien existe mimic/live_demo.py para
medir FPS reales sin la latencia del navegador.

Uso:
    python -m app.app
"""
import time
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
_cliente_grounding = None
_grounding_intentado = False


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


def _obtener_cliente_grounding():
    """Carga perezosa de LocateAnything-3B. Si el hardware no es
    compatible (sin GPU Ampere+), se recuerda el fallo una sola vez en
    vez de reintentar en cada consulta, y se devuelve None para que la
    UI muestre un mensaje claro en lugar de fallar en cada clic."""
    global _cliente_grounding, _grounding_intentado
    if _grounding_intentado:
        return _cliente_grounding
    _grounding_intentado = True
    from ask.grounding import ClienteLocateAnything, ErrorHardwareNoCompatible
    try:
        _cliente_grounding = ClienteLocateAnything()
    except ErrorHardwareNoCompatible:
        _cliente_grounding = None
    return _cliente_grounding


def procesar_video_ask(ruta_video):
    if ruta_video is None:
        return [], "Sube un video primero."
    from ask.ingest import muestrear_frames_de_video
    from ask.timeline import construir_timeline

    _inicializar()
    frames_muestreados = muestrear_frames_de_video(ruta_video, fps_muestreo=8.0)
    # tamano_ventana=15: el mismo valor que D3_ask_video.ipynb encontro
    # mejor contra la verdad de terreno del video de prueba (seccion 2
    # del notebook) -- con 5 el timeline queda fragmentado en decenas
    # de micro-eventos por el parpadeo del clasificador entre frames.
    eventos = construir_timeline(frames_muestreados, _detector, _modelo, tamano_ventana=15)
    resumen = "\n".join(f"{e['start_time']:.1f}s-{e['end_time']:.1f}s: {e['type']}" for e in eventos)
    return eventos, (resumen or "No se detectaron eventos.")


def consultar_ask(consulta, eventos):
    if not eventos:
        return "Primero procesa un video para construir el timeline."
    cliente = _obtener_cliente_grounding()
    if cliente is None:
        return (
            "LocateAnything-3B no esta disponible en este entorno (requiere GPU "
            "Ampere o superior, ver docs/superpowers/specs/2026-07-05-ask-design.md). "
            "Corre este modo en Colab con GPU."
        )
    from ask.query_engine import responder_consulta
    resultado = responder_consulta(consulta, eventos, cliente)
    if resultado is None:
        return "No se encontro ninguna coincidencia para esa consulta."
    return f"{resultado['type']} entre {resultado['start_time']:.1f}s y {resultado['end_time']:.1f}s"


def _frame_en_vivo_ask(imagen, estado):
    """Actualiza el timeline en vivo un frame a la vez. El estado
    (suavizador, evento abierto, eventos cerrados) vive en gr.State,
    que Gradio mantiene por sesion entre llamadas."""
    if imagen is None:
        return estado, "Esperando imagen..."
    _inicializar()
    if estado is None:
        estado = {"suavizador": SuavizadorTemporal(), "evento_abierto": None, "eventos": []}

    resultado = procesar_frame(imagen[:, :, ::-1], _detector, _modelo, timestamp=time.time())
    if resultado["etiqueta_pose"] is None:
        return estado, f"Nadie detectado | Eventos cerrados: {len(estado['eventos'])}"

    etiqueta_estable = estado["suavizador"].actualizar(resultado["etiqueta_pose"])
    evento_abierto = estado["evento_abierto"]

    if evento_abierto is None:
        estado["evento_abierto"] = {
            "type": etiqueta_estable, "start_time": resultado["timestamp"], "frames": [imagen],
        }
    elif evento_abierto["type"] != etiqueta_estable:
        frames = evento_abierto["frames"]
        estado["eventos"].append({
            "type": evento_abierto["type"],
            "start_time": evento_abierto["start_time"],
            "end_time": resultado["timestamp"],
            "duration_s": resultado["timestamp"] - evento_abierto["start_time"],
            "frame_representativo": frames[len(frames) // 2],
        })
        estado["evento_abierto"] = {
            "type": etiqueta_estable, "start_time": resultado["timestamp"], "frames": [imagen],
        }
    else:
        evento_abierto["frames"].append(imagen)

    return estado, f"Actual: {etiqueta_estable} | Eventos cerrados: {len(estado['eventos'])}"


def consultar_ask_en_vivo(consulta, estado):
    if estado is None or not estado["eventos"]:
        return "Todavia no hay eventos acumulados en esta sesion en vivo."
    return consultar_ask(consulta, estado["eventos"])


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
        with gr.Tab("ASK"):
            gr.Markdown(
                "Video o camara en vivo: construye una linea temporal de poses/gestos "
                "y responde una consulta en lenguaje natural con evidencia. "
                "El grounding con LocateAnything-3B requiere GPU (Colab)."
            )
            eventos_video_state = gr.State([])
            with gr.Tab("Video"):
                entrada_video_ask = gr.Video(label="Video a analizar")
                boton_procesar = gr.Button("Construir timeline")
                resumen_timeline = gr.Textbox(label="Eventos detectados", lines=6)
                consulta_video = gr.Textbox(label="Consulta en lenguaje natural")
                boton_consultar_video = gr.Button("Consultar")
                respuesta_video = gr.Textbox(label="Respuesta")

                boton_procesar.click(
                    procesar_video_ask, inputs=entrada_video_ask,
                    outputs=[eventos_video_state, resumen_timeline],
                )
                boton_consultar_video.click(
                    consultar_ask, inputs=[consulta_video, eventos_video_state],
                    outputs=respuesta_video,
                )
            with gr.Tab("En vivo"):
                estado_en_vivo = gr.State(None)
                entrada_en_vivo = gr.Image(sources=["webcam"], streaming=True)
                estado_texto_en_vivo = gr.Textbox(label="Estado del timeline en vivo")
                entrada_en_vivo.stream(
                    _frame_en_vivo_ask, inputs=[entrada_en_vivo, estado_en_vivo],
                    outputs=[estado_en_vivo, estado_texto_en_vivo],
                )
                consulta_en_vivo = gr.Textbox(label="Consulta en lenguaje natural")
                boton_consultar_en_vivo = gr.Button("Consultar")
                respuesta_en_vivo = gr.Textbox(label="Respuesta")
                boton_consultar_en_vivo.click(
                    consultar_ask_en_vivo, inputs=[consulta_en_vivo, estado_en_vivo],
                    outputs=respuesta_en_vivo,
                )
    return demo


if __name__ == "__main__":
    construir_demo().launch()
