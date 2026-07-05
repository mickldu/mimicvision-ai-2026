"""Indice de similitud coseno sobre los vectores de la galeria.

Con ~290 imagenes, una busqueda exacta por fuerza bruta es instantanea:
no hace falta FAISS ni ninguna estructura aproximada a esta escala, y
evitarlo ahorra una dependencia pesada con friccion de instalacion en
Windows. El docx permite explicitamente esta alternativa."""
import numpy as np


def _normalizar(vectores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / (normas + 1e-9)


class IndiceSimilitud:
    def __init__(self, vectores: np.ndarray, metadatos: list[dict]):
        self._vectores_normalizados = _normalizar(np.asarray(vectores, dtype=np.float64))
        self._metadatos = metadatos

    def buscar(self, vector_consulta: np.ndarray, k: int = 5) -> list[dict]:
        consulta = np.asarray(vector_consulta, dtype=np.float64).reshape(1, -1)
        consulta_normalizada = _normalizar(consulta)[0]
        scores = self._vectores_normalizados @ consulta_normalizada
        indices_top_k = np.argsort(-scores)[:k]

        resultados = []
        for i in indices_top_k:
            resultado = dict(self._metadatos[i])
            resultado["score"] = float(scores[i])
            resultados.append(resultado)
        return resultados
