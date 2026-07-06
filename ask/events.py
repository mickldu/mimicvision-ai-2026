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
