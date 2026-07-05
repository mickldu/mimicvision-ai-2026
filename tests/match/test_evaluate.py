import pytest

from match.evaluate import mrr, recall_at_k


def _resultado(etiqueta):
    return {"etiqueta_pose": etiqueta}


def test_recall_at_1_cuenta_solo_el_primer_resultado():
    resultados_por_consulta = [
        [_resultado("saludo"), _resultado("zen")],
        [_resultado("zen"), _resultado("saludo")],
    ]
    clases_verdaderas = ["saludo", "saludo"]

    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=1) == 0.5


def test_recall_at_5_es_mas_permisivo_que_recall_at_1():
    resultados_por_consulta = [[_resultado("zen"), _resultado("saludo")]]
    clases_verdaderas = ["saludo"]

    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=1) == 0.0
    assert recall_at_k(resultados_por_consulta, clases_verdaderas, k=5) == 1.0


def test_mrr_premia_que_el_correcto_este_mas_arriba():
    resultados_por_consulta = [
        [_resultado("saludo")],
        [_resultado("zen"), _resultado("saludo")],
    ]
    clases_verdaderas = ["saludo", "saludo"]

    valor = mrr(resultados_por_consulta, clases_verdaderas)

    assert valor == pytest.approx((1.0 + 0.5) / 2)
