from types import SimpleNamespace

import numpy as np
import pytest

from mimic.features import (
    angulo,
    calcular_bbox_persona,
    construir_vector_features,
    distancia,
    NOMBRES_FEATURES,
)


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


def test_calcular_bbox_persona_encierra_los_landmarks_con_margen():
    # Los landmarks ocupan el rectangulo normalizado x=[0.3, 0.7], y=[0.15, 0.6]
    pose = {
        11: _punto(0.4, 0.3), 12: _punto(0.6, 0.3),
        13: _punto(0.35, 0.45), 14: _punto(0.65, 0.45),
        15: _punto(0.3, 0.6), 16: _punto(0.7, 0.6),
        0: _punto(0.5, 0.15),
    }
    ancho_imagen, alto_imagen = 200, 100

    x, y, w, h = calcular_bbox_persona(pose, ancho_imagen, alto_imagen, margen=0.0)

    # sin margen, el rectangulo debe coincidir exactamente con el rango de landmarks
    assert x == pytest.approx(0.3 * ancho_imagen, abs=1)
    assert y == pytest.approx(0.15 * alto_imagen, abs=1)
    assert x + w == pytest.approx(0.7 * ancho_imagen, abs=1)
    assert y + h == pytest.approx(0.6 * alto_imagen, abs=1)


def test_calcular_bbox_persona_no_se_sale_de_la_imagen_con_margen():
    # Landmarks pegados al borde: el margen no debe empujar la caja fuera del frame
    pose = {
        11: _punto(0.02, 0.02), 12: _punto(0.98, 0.02),
        13: _punto(0.02, 0.5), 14: _punto(0.98, 0.5),
        15: _punto(0.0, 0.9), 16: _punto(1.0, 0.9),
        0: _punto(0.5, 0.0),
    }
    ancho_imagen, alto_imagen = 300, 300

    x, y, w, h = calcular_bbox_persona(pose, ancho_imagen, alto_imagen, margen=0.15)

    assert x >= 0
    assert y >= 0
    assert x + w <= ancho_imagen
    assert y + h <= alto_imagen
