import pandas as pd

from match.gallery import cargar_galeria_y_consultas


def test_separa_galeria_de_consultas_por_split(tmp_path):
    ruta = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"sample_id": "a", "clase": "saludo", "split": "train"},
        {"sample_id": "b", "clase": "saludo", "split": "val"},
        {"sample_id": "c", "clase": "saludo", "split": "test"},
        {"sample_id": "d", "clase": "zen", "split": "test"},
    ]).to_csv(ruta, index=False)

    galeria, consultas = cargar_galeria_y_consultas(str(ruta))

    assert set(galeria["sample_id"]) == {"a", "b"}
    assert set(consultas["sample_id"]) == {"c", "d"}
