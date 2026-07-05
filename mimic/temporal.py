"""Suaviza predicciones frame a frame para evitar que la etiqueta
parpadee entre clases parecidas de un frame a otro."""
from collections import deque, Counter


class SuavizadorTemporal:
    def __init__(self, tamano_ventana: int = 7):
        self._ventana = deque(maxlen=tamano_ventana)

    def actualizar(self, etiqueta: str) -> str:
        self._ventana.append(etiqueta)
        conteo = Counter(self._ventana)
        return conteo.most_common(1)[0][0]
