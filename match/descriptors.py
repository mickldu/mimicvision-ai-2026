"""Descriptores clasicos de imagen (ruta A del ablation): HOG y LBP.

Toda imagen se redimensiona primero a un tamano fijo -- si no, HOG
devolveria vectores de longitud distinta segun el tamano original de
cada foto, y no se podrian comparar por similitud coseno."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

TAMANO_FIJO = (128, 128)
LBP_PUNTOS = 24
LBP_RADIO = 3


def _a_gris_redimensionado(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, TAMANO_FIJO)


def extraer_hog(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = _a_gris_redimensionado(imagen_bgr)
    return hog(gris, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def extraer_lbp_histograma(imagen_bgr: np.ndarray) -> np.ndarray:
    gris = _a_gris_redimensionado(imagen_bgr)
    patrones = local_binary_pattern(gris, LBP_PUNTOS, LBP_RADIO, method="uniform")
    histograma, _ = np.histogram(
        patrones, bins=np.arange(0, LBP_PUNTOS + 3), range=(0, LBP_PUNTOS + 2)
    )
    return histograma.astype(np.float64) / (histograma.sum() + 1e-9)
