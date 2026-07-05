"""Funciones geometricas puras sobre puntos 2D normalizados de MediaPipe.

Se trabaja siempre con arreglos numpy de forma (2,) -- x, y en
coordenadas normalizadas de la imagen (0 a 1), tal como las entrega
HolisticLandmarker. No dependen de mediapipe directamente, para poder
probarlas con puntos inventados.
"""
import numpy as np


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo en grados formado en el vertice b por los segmentos b-a y b-c."""
    ba = a - b
    bc = c - b
    coseno = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    coseno = np.clip(coseno, -1.0, 1.0)
    return float(np.degrees(np.arccos(coseno)))


def distancia(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


HOMBRO_IZQ, HOMBRO_DER = 11, 12
CODO_IZQ, CODO_DER = 13, 14
MUNECA_IZQ, MUNECA_DER = 15, 16
NARIZ = 0

NOMBRES_FEATURES = [
    "angulo_codo_izq",
    "angulo_codo_der",
    "angulo_hombro_izq",
    "angulo_hombro_der",
    "inclinacion_cabeza",
    "distancia_manos",
    "distancia_mano_menton",
    "distancia_mano_hombro_contrario",
    "altura_manos_relativa",
    "asimetria_manos",
]


def _p(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y])


def construir_vector_features(pose: dict, menton) -> list[float]:
    """Convierte landmarks crudos en el vector de 10 features de MIMIC.

    'pose' es un diccionario {indice: landmark} con al menos hombros,
    codos, munecas y nariz -- lo que HolisticLandmarker entrega siempre
    que el torso superior sea visible, sin importar si es una foto de
    cuerpo completo o un plano de medio cuerpo como en HaGRID.
    """
    hombro_izq, hombro_der = _p(pose[HOMBRO_IZQ]), _p(pose[HOMBRO_DER])
    codo_izq, codo_der = _p(pose[CODO_IZQ]), _p(pose[CODO_DER])
    muneca_izq, muneca_der = _p(pose[MUNECA_IZQ]), _p(pose[MUNECA_DER])
    nariz = _p(pose[NARIZ])
    menton_p = _p(menton)

    ancho_hombros = distancia(hombro_izq, hombro_der) + 1e-9
    hombro_medio = (hombro_izq + hombro_der) / 2
    vertical = np.array([0.0, 1.0])

    angulo_codo_izq = angulo(hombro_izq, codo_izq, muneca_izq)
    angulo_codo_der = angulo(hombro_der, codo_der, muneca_der)
    angulo_hombro_izq = angulo(codo_izq, hombro_izq, hombro_izq + vertical)
    angulo_hombro_der = angulo(codo_der, hombro_der, hombro_der + vertical)
    inclinacion_cabeza = angulo(nariz, hombro_medio, hombro_medio + vertical)

    distancia_manos = distancia(muneca_izq, muneca_der) / ancho_hombros
    distancia_mano_menton = min(
        distancia(muneca_izq, menton_p), distancia(muneca_der, menton_p)
    ) / ancho_hombros
    distancia_mano_hombro_contrario = min(
        distancia(muneca_izq, hombro_der), distancia(muneca_der, hombro_izq)
    ) / ancho_hombros
    altura_manos_relativa = (
        (muneca_izq[1] + muneca_der[1]) / 2 - hombro_medio[1]
    ) / ancho_hombros
    asimetria_manos = abs(muneca_izq[1] - muneca_der[1]) / ancho_hombros

    return [
        angulo_codo_izq,
        angulo_codo_der,
        angulo_hombro_izq,
        angulo_hombro_der,
        inclinacion_cabeza,
        distancia_manos,
        distancia_mano_menton,
        distancia_mano_hombro_contrario,
        altura_manos_relativa,
        asimetria_manos,
    ]
