from pathlib import Path

import cv2
import numpy as np

from mimic.capture import leer_frames_de_video


def _crear_video_de_prueba(ruta: Path, num_frames: int = 5):
    escritor = cv2.VideoWriter(
        str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64)
    )
    for _ in range(num_frames):
        escritor.write(np.zeros((64, 64, 3), dtype=np.uint8))
    escritor.release()


def test_lee_todos_los_frames_de_un_video(tmp_path):
    ruta_video = tmp_path / "prueba.mp4"
    _crear_video_de_prueba(ruta_video, num_frames=5)

    frames = list(leer_frames_de_video(str(ruta_video)))

    assert len(frames) == 5
    assert frames[0].shape == (64, 64, 3)
