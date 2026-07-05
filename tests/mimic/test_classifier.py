import numpy as np

from mimic.classifier import cargar_modelo, entrenar_y_seleccionar, guardar_modelo


def test_entrena_y_selecciona_el_mejor_modelo_por_f1_macro():
    # dos clases perfectamente separables en el espacio de features
    rng = np.random.default_rng(42)
    clase_a = rng.normal(loc=0.0, scale=0.1, size=(30, 10))
    clase_b = rng.normal(loc=5.0, scale=0.1, size=(30, 10))
    X = np.vstack([clase_a, clase_b])
    y = ["neutral"] * 30 + ["zen"] * 30

    resultado = entrenar_y_seleccionar(X, y)

    assert resultado.modelo is not None
    assert resultado.nombre_modelo in ("svm", "random_forest")
    assert resultado.f1_macro_validacion > 0.9


def test_guarda_y_carga_el_modelo(tmp_path):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 10)), rng.normal(5, 0.1, (20, 10))])
    y = ["a"] * 20 + ["b"] * 20
    resultado = entrenar_y_seleccionar(X, y)

    ruta = tmp_path / "modelo.joblib"
    guardar_modelo(resultado.modelo, ruta)
    modelo_cargado = cargar_modelo(ruta)

    prediccion = modelo_cargado.predict(X[:1])
    assert prediccion[0] in ("a", "b")
