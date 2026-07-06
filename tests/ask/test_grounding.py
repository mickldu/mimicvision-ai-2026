import pytest

from ask.grounding import ClienteLocateAnything, ErrorHardwareNoCompatible, parsear_cajas


def test_parsear_cajas_con_formato_separado_por_comas():
    texto = "Encontrado: <box>100,200,300,400</box>"
    assert parsear_cajas(texto, ancho=1000, alto=1000) == [(100, 200, 300, 400)]


def test_parsear_cajas_con_tags_individuales_por_coordenada():
    # La documentacion del modelo no deja 100% claro el separador exacto
    # entre coordenadas -- este test cubre el formato alternativo donde
    # cada numero va en su propia sub-etiqueta.
    texto = "<box><x1>500</x1><y1>0</y1><x2>1000</x2><y2>500</y2></box>"
    assert parsear_cajas(texto, ancho=200, alto=100) == [(100, 0, 200, 50)]


def test_parsear_cajas_ignora_texto_sin_etiqueta_box():
    assert parsear_cajas("No se encontro ninguna coincidencia.", ancho=100, alto=100) == []


def test_parsear_cajas_con_multiples_cajas():
    texto = "<box>0,0,500,500</box> y tambien <box>500,500,1000,1000</box>"
    assert parsear_cajas(texto, ancho=100, alto=100) == [(0, 0, 50, 50), (50, 50, 100, 100)]


def test_cliente_lanza_error_claro_si_no_hay_gpu_compatible():
    with pytest.raises(ErrorHardwareNoCompatible):
        ClienteLocateAnything()
