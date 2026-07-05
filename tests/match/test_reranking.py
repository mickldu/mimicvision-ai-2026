import numpy as np

from match.reranking import reordenar_por_geometria


def test_reordena_poniendo_primero_al_mas_parecido_geometricamente():
    candidatos = [
        {"image_id": "a"},
        {"image_id": "b"},
    ]
    vector_consulta = np.array([1.0, 0.0])
    vectores_por_id = {
        "a": np.array([0.0, 1.0]),   # geometria opuesta a la consulta
        "b": np.array([0.9, 0.1]),   # geometria parecida a la consulta
    }

    reordenados = reordenar_por_geometria(candidatos, vector_consulta, vectores_por_id)

    assert [c["image_id"] for c in reordenados] == ["b", "a"]
