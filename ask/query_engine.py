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
