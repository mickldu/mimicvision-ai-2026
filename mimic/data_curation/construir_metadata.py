"""Recorre data/images/<clase>/ y construye data/metadata.csv, el
inventario oficial del dataset que exige el protocolo de datos del
proyecto: cada muestra con su clase, fuente, licencia y split.

El split es estratificado por clase (70/15/15) y queda grabado en el
CSV para que cualquier re-ejecucion del notebook use exactamente la
misma particion -- sin esto, las metricas no serian comparables entre
corridas ni entre companeros del equipo.

Uso:
    python -m mimic.data_curation.construir_metadata
"""
import csv
import random
from pathlib import Path

from mimic.data_curation.manifiesto import CLASES

RAIZ_DATOS = Path("data/images")
RUTA_METADATA = Path("data/metadata.csv")
RUTA_REGISTRO_CC = Path("data/raw/cc_registro.csv")

SEMILLA = 42
FRACCION_TRAIN = 0.70
FRACCION_VAL = 0.15
# el resto (15%) es test

LICENCIA_HAGRID = "CC-BY-SA-4.0"
URL_HAGRID = "https://huggingface.co/datasets/cj-mills/hagrid-classification-512p-no-gesture-150k"


def _cargar_registro_cc() -> dict:
    """Indexa el registro de curacion CC por nombre de archivo para
    recuperar la licencia y el origen de cada foto curada."""
    registro = {}
    if RUTA_REGISTRO_CC.exists():
        with open(RUTA_REGISTRO_CC, encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                registro[fila["archivo"]] = fila
    return registro


def _asignar_splits(cantidad: int, aleatorio: random.Random) -> list[str]:
    """Devuelve la lista de splits para una clase, mezclada de forma
    reproducible. Se garantiza al menos 1 muestra en val y en test
    incluso en clases pequenas."""
    n_train = max(1, round(cantidad * FRACCION_TRAIN))
    n_val = max(1, round(cantidad * FRACCION_VAL))
    n_test = max(1, cantidad - n_train - n_val)
    # si el redondeo se pasa del total, se recorta train que es el mas grande
    exceso = (n_train + n_val + n_test) - cantidad
    n_train -= exceso
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    aleatorio.shuffle(splits)
    return splits


def main():
    registro_cc = _cargar_registro_cc()
    aleatorio = random.Random(SEMILLA)
    filas = []

    for definicion in CLASES:
        carpeta = RAIZ_DATOS / definicion.nombre
        rutas = sorted(carpeta.glob("*.jpg"))
        if not rutas:
            print(f"AVISO: {definicion.nombre} no tiene imagenes todavia")
            continue

        splits = _asignar_splits(len(rutas), aleatorio)
        for ruta, split in zip(rutas, splits):
            if definicion.fuente == "hagrid":
                licencia, url_origen = LICENCIA_HAGRID, URL_HAGRID
                condicion = "hagrid_multiples_sujetos"
            else:
                info = registro_cc.get(ruta.name, {})
                licencia = info.get("licencia", "desconocida")
                url_origen = info.get("pagina_origen", info.get("url_origen", ""))
                condicion = info.get("autor", "desconocido")

            filas.append({
                "sample_id": ruta.stem,
                "clase": definicion.nombre,
                "tipo": definicion.tipo,
                "fuente": definicion.fuente,
                "licencia": licencia,
                "url_origen": url_origen,
                "participante_o_condicion": condicion,
                "split": split,
                "ruta": str(ruta).replace("\\", "/"),
            })

    RUTA_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_METADATA, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"{len(filas)} muestras registradas en {RUTA_METADATA}")
    for definicion in CLASES:
        del_total = [f for f in filas if f["clase"] == definicion.nombre]
        if del_total:
            print(f"  {definicion.nombre}: {len(del_total)}")


if __name__ == "__main__":
    main()
