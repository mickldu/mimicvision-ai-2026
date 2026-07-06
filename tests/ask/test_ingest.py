from pathlib import Path

import cv2
import numpy as np
import pytest

from ask.ingest import muestrear_frames_de_video, muestrear_frames_en_vivo


def _crear_video_de_prueba(ruta: Path, num_frames: int = 50, fps: float = 25.0):
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for _ in range(num_frames):
        escritor.write(np.zeros((32, 32, 3), dtype=np.uint8))
    escritor.release()


def test_muestrear_frames_de_video_reduce_la_tasa_a_lo_pedido(tmp_path):
    ruta_video = tmp_path / "prueba.mp4"
    _crear_video_de_prueba(ruta_video, num_frames=50, fps=25.0)

    muestras = list(muestrear_frames_de_video(str(ruta_video), fps_muestreo=5.0))

    # 50 frames a 25fps duran 2s; a 5fps de muestreo esperamos ~10 muestras
    assert len(muestras) == 10
    assert muestras[0][0] == pytest.approx(0.0)
    assert muestras[1][0] == pytest.approx(0.2, abs=1e-6)


def test_muestrear_frames_en_vivo_pasa_cada_frame_con_timestamp():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]

    resultado = list(muestrear_frames_en_vivo(iter(frames)))

    assert len(resultado) == 3
    assert all(isinstance(marca_tiempo, float) for marca_tiempo, _ in resultado)
    assert resultado[0][1] is frames[0]
