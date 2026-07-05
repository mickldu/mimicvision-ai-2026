from pathlib import Path

import cv2
import numpy as np

from match.embeddings import ExtractorSigLIP2

FIXTURE = Path(__file__).parent.parent / "fixtures" / "persona_prueba.jpg"


def test_extraer_devuelve_vector_768_no_nulo():
    extractor = ExtractorSigLIP2()
    imagen = cv2.imread(str(FIXTURE))

    vector = extractor.extraer(imagen)

    assert vector.shape == (768,)
    assert not np.isnan(vector).any()
