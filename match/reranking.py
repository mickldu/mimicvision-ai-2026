"""Re-ranking de un Top-K por similitud geometrica (ruta B), para el
experimento E-D del docx: combinar semantica visual (embeddings) con
geometria de pose para ver si mejora el Recall@1."""
import numpy as np


def _similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def reordenar_por_geometria(
    candidatos: list[dict],
    vector_geometrico_consulta: np.ndarray,
    vectores_geometricos_por_id: dict,
) -> list[dict]:
    def puntaje(candidato):
        vector_candidato = vectores_geometricos_por_id[candidato["image_id"]]
        return _similitud_coseno(vector_geometrico_consulta, vector_candidato)

    return sorted(candidatos, key=puntaje, reverse=True)
