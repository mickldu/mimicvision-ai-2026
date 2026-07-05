from pathlib import Path

import cv2

from mimic.landmarks import DetectorHolistic

FIXTURE = Path(__file__).parent.parent / "fixtures" / "persona_prueba.jpg"


def test_detecta_landmarks_de_pose_y_rostro_en_foto_real():
    detector = DetectorHolistic()
    frame = cv2.imread(str(FIXTURE))
    resultado = detector.procesar(frame)

    assert resultado.pose is not None
    assert 11 in resultado.pose  # hombro izquierdo detectado
    assert resultado.menton is not None


def test_procesa_imagenes_de_tamanos_distintos_consecutivas():
    # Regresion: el HolisticLandmarker de mediapipe 0.10.35 guarda estado
    # interno de segmentacion y crashea si dos imagenes consecutivas
    # tienen dimensiones diferentes. El letterbox interno debe evitarlo.
    detector = DetectorHolistic()
    frame_grande = cv2.imread(str(FIXTURE))
    frame_chico = cv2.resize(frame_grande, (300, 200))

    detector.procesar(frame_grande)
    detector.procesar(frame_chico)  # antes esto lanzaba RuntimeError
    resultado = detector.procesar(frame_grande)

    assert resultado.pose is not None
