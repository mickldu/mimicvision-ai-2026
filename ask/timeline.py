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
