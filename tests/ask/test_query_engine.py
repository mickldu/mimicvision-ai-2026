from unittest.mock import MagicMock

import numpy as np

from ask.query_engine import responder_consulta


def _evento(tipo, inicio, fin, frame):
    return {"type": tipo, "start_time": inicio, "end_time": fin, "duration_s": fin - inicio, "frame_representativo": frame}


def test_responder_consulta_devuelve_el_primer_evento_con_coincidencia():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    eventos_preliminares = [
        _evento("neutral", 0.0, 1.0, frame),
        _evento("brazos_cruzados", 1.0, 2.0, frame),
    ]
    cliente_falso = MagicMock()
    cliente_falso.localizar.side_effect = [
        {"consulta": "arms crossed", "cajas": []},
        {"consulta": "arms crossed", "cajas": [(1, 2, 3, 4)]},
    ]

    resultado = responder_consulta("arms crossed", eventos_preliminares, cliente_falso)

    assert resultado is not None
    assert resultado["type"] == "brazos_cruzados"
    assert resultado["source"] == "MIMIC+LocateAnything"


def test_responder_consulta_devuelve_none_si_nada_coincide():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    eventos_preliminares = [_evento("neutral", 0.0, 1.0, frame)]
    cliente_falso = MagicMock()
    cliente_falso.localizar.return_value = {"consulta": "x", "cajas": []}

    assert responder_consulta("algo que no existe", eventos_preliminares, cliente_falso) is None
