from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mimic.pipeline import procesar_frame


def test_procesar_frame_devuelve_perception_result_con_etiqueta():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(
        pose={
            11: SimpleNamespace(x=0.4, y=0.3), 12: SimpleNamespace(x=0.6, y=0.3),
            13: SimpleNamespace(x=0.35, y=0.45), 14: SimpleNamespace(x=0.65, y=0.45),
            15: SimpleNamespace(x=0.3, y=0.6), 16: SimpleNamespace(x=0.7, y=0.6),
            0: SimpleNamespace(x=0.5, y=0.15),
        },
        menton=SimpleNamespace(x=0.5, y=0.2),
    )
    modelo_falso = MagicMock()
    modelo_falso.predict.return_value = ["neutral"]
    modelo_falso.predict_proba.return_value = np.array([[0.9, 0.1]])
    modelo_falso.classes_ = ["neutral", "zen"]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resultado = procesar_frame(frame, detector_falso, modelo_falso, timestamp=1.5)

    assert resultado["etiqueta_pose"] == "neutral"
    assert resultado["confianza"] == 0.9
    assert resultado["timestamp"] == 1.5
    assert resultado["frame"] is frame
    # el bbox debe ser una caja valida dentro del frame de 100x100
    x, y, w, h = resultado["bbox_persona"]
    assert w > 0 and h > 0
    assert 0 <= x and x + w <= 100
    assert 0 <= y and y + h <= 100


def test_procesar_frame_sin_persona_devuelve_etiqueta_none():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(pose=None, menton=None)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    resultado = procesar_frame(frame, detector_falso, modelo=MagicMock(), timestamp=0.0)

    assert resultado["etiqueta_pose"] is None
    assert resultado["confianza"] == 0.0
    assert resultado["bbox_persona"] is None
