from types import SimpleNamespace

import numpy as np
import pytest

from mimic.features import angulo, construir_vector_features, distancia, NOMBRES_FEATURES


def test_angulo_90_grados():
    # vertice en el origen, un brazo hacia arriba, otro hacia la derecha
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angulo(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_angulo_180_grados():
    # los tres puntos alineados: el angulo en b debe ser 180
    a = np.array([-1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angulo(a, b, c) == pytest.approx(180.0, abs=0.1)


def test_distancia_pitagoras():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert distancia(a, b) == pytest.approx(5.0, abs=1e-6)


def _punto(x, y):
    return SimpleNamespace(x=x, y=y)


def test_construir_vector_features_devuelve_10_valores_nombrados():
    # Postura simetrica e inventada: brazos hacia abajo, manos separadas
    pose = {
        11: _punto(0.4, 0.3),  # hombro izq
        12: _punto(0.6, 0.3),  # hombro der
        13: _punto(0.35, 0.45),  # codo izq
        14: _punto(0.65, 0.45),  # codo der
        15: _punto(0.3, 0.6),  # muneca izq
        16: _punto(0.7, 0.6),  # muneca der
        0: _punto(0.5, 0.15),  # nariz
    }
    menton = _punto(0.5, 0.2)

    vector = construir_vector_features(pose, menton)

    assert len(vector) == 10
    assert len(NOMBRES_FEATURES) == 10
    assert all(isinstance(v, float) for v in vector)


def test_construir_vector_features_lanza_error_si_falta_un_punto_clave():
    pose_incompleta = {11: _punto(0.4, 0.3)}  # faltan los demas puntos
    with pytest.raises(KeyError):
        construir_vector_features(pose_incompleta, _punto(0.5, 0.2))
