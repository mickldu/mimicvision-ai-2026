import cv2
import numpy as np
import pytest

from match.descriptors import extraer_hog, extraer_lbp_histograma

FRAME_PRUEBA = np.random.default_rng(0).integers(0, 255, (200, 150, 3), dtype=np.uint8)


def test_extraer_hog_devuelve_vector_de_longitud_fija():
    vector = extraer_hog(FRAME_PRUEBA)
    assert vector.shape == (1764,)


def test_extraer_hog_es_igual_sin_importar_el_tamano_original():
    otro_tamano = cv2.resize(FRAME_PRUEBA, (400, 300))
    v1 = extraer_hog(FRAME_PRUEBA)
    v2 = extraer_hog(otro_tamano)
    assert v1.shape == v2.shape


def test_extraer_lbp_histograma_suma_1():
    histograma = extraer_lbp_histograma(FRAME_PRUEBA)
    assert histograma.shape == (26,)
    assert histograma.sum() == pytest.approx(1.0, abs=1e-6)
