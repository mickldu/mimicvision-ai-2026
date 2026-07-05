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
