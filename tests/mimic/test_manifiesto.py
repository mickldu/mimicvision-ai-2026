from mimic.data_curation.manifiesto import CLASES


def test_manifiesto_tiene_10_clases():
    assert len(CLASES) == 10


def test_cada_clase_declara_su_tipo_y_fuente():
    tipos_validos = {"gesto", "pose"}
    fuentes_validas = {"hagrid", "fotos_cc"}
    for clase in CLASES:
        assert clase.tipo in tipos_validos
        assert clase.fuente in fuentes_validas


def test_las_dos_clases_de_hagrid_estan_declaradas():
    nombres_hagrid = {c.nombre for c in CLASES if c.fuente == "hagrid"}
    assert nombres_hagrid == {"saludo", "pulgar_arriba"}
