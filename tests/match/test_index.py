import numpy as np
import pytest

from match.index import IndiceSimilitud


def test_buscar_devuelve_el_vector_identico_primero():
    vectores = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ])
    metadatos = [
        {"image_id": "a", "etiqueta_pose": "saludo"},
        {"image_id": "b", "etiqueta_pose": "zen"},
        {"image_id": "c", "etiqueta_pose": "saludo"},
    ]
    indice = IndiceSimilitud(vectores, metadatos)

    resultados = indice.buscar(np.array([1.0, 0.0, 0.0]), k=2)

    assert resultados[0]["image_id"] == "a"
    assert resultados[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert resultados[1]["image_id"] == "c"


def test_buscar_respeta_k():
    vectores = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    metadatos = [{"image_id": str(i)} for i in range(3)]
    indice = IndiceSimilitud(vectores, metadatos)

    resultados = indice.buscar(np.array([1.0, 0.0]), k=1)

    assert len(resultados) == 1
