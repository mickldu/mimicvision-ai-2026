"""Metricas de recuperacion sobre resultados Top-K: Recall@K y MRR.

El criterio de relevancia es simple y explicito: un resultado cuenta
como correcto si su etiqueta_pose coincide con la clase real de la
consulta -- no hay anotacion manual de relevancia mas fina que esa."""


def recall_at_k(resultados_por_consulta: list[list[dict]], clases_verdaderas: list[str], k: int) -> float:
    aciertos = 0
    for resultados, clase_real in zip(resultados_por_consulta, clases_verdaderas):
        clases_top_k = [r["etiqueta_pose"] for r in resultados[:k]]
        if clase_real in clases_top_k:
            aciertos += 1
    return aciertos / len(clases_verdaderas)


def mrr(resultados_por_consulta: list[list[dict]], clases_verdaderas: list[str]) -> float:
    reciprocos = []
    for resultados, clase_real in zip(resultados_por_consulta, clases_verdaderas):
        rango = None
        for posicion, resultado in enumerate(resultados, start=1):
            if resultado["etiqueta_pose"] == clase_real:
                rango = posicion
                break
        reciprocos.append(1.0 / rango if rango else 0.0)
    return sum(reciprocos) / len(reciprocos)
