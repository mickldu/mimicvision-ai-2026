from mimic.temporal import SuavizadorTemporal


def test_devuelve_la_clase_mayoritaria_en_la_ventana():
    suavizador = SuavizadorTemporal(tamano_ventana=5)
    etiquetas = ["neutral", "neutral", "zen", "neutral", "zen"]
    resultado = None
    for etiqueta in etiquetas:
        resultado = suavizador.actualizar(etiqueta)
    assert resultado == "neutral"


def test_ventana_deslizante_olvida_predicciones_viejas():
    suavizador = SuavizadorTemporal(tamano_ventana=3)
    for etiqueta in ["neutral", "neutral", "neutral"]:
        suavizador.actualizar(etiqueta)
    # las siguientes 3 predicciones deberian desplazar del todo a "neutral"
    resultado = None
    for etiqueta in ["zen", "zen", "zen"]:
        resultado = suavizador.actualizar(etiqueta)
    assert resultado == "zen"
