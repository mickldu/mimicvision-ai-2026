from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ask.timeline import construir_timeline


def _pose_valida():
    return {
        11: SimpleNamespace(x=0.4, y=0.3), 12: SimpleNamespace(x=0.6, y=0.3),
        13: SimpleNamespace(x=0.35, y=0.45), 14: SimpleNamespace(x=0.65, y=0.45),
        15: SimpleNamespace(x=0.3, y=0.6), 16: SimpleNamespace(x=0.7, y=0.6),
        0: SimpleNamespace(x=0.5, y=0.15),
    }


def test_construir_timeline_agrupa_frames_consecutivos_de_la_misma_clase():
    detector_falso = MagicMock()
    detector_falso.procesar.return_value = SimpleNamespace(
        pose=_pose_valida(), menton=SimpleNamespace(x=0.5, y=0.2)
    )

    modelo_falso = MagicMock()
    modelo_falso.classes_ = ["neutral", "zen"]
    secuencia = ["neutral", "neutral", "neutral", "zen", "zen", "zen"]
    modelo_falso.predict.side_effect = [[e] for e in secuencia]
    modelo_falso.predict_proba.side_effect = [
        np.array([[0.9, 0.1]]) if e == "neutral" else np.array([[0.1, 0.9]]) for e in secuencia
    ]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames_muestreados = [(i * 0.1, frame) for i in range(6)]

    eventos = construir_timeline(frames_muestreados, detector_falso, modelo_falso, tamano_ventana=1)

    assert len(eventos) == 2
    assert eventos[0]["type"] == "neutral"
    assert eventos[0]["start_time"] == pytest.approx(0.0)
    assert eventos[0]["end_time"] == pytest.approx(0.3)
    assert eventos[1]["type"] == "zen"
    assert eventos[1]["start_time"] == pytest.approx(0.3)
    assert "frame_representativo" in eventos[0]
