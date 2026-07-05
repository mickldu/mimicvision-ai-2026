"""Descarga un subconjunto de HaGRID para las clases saludo (palm) y
pulgar_arriba (like), sin bajar el dataset completo de 3.8 GB.

Se apoya en la API publica datasets-server de Hugging Face sobre el
dataset cj-mills/hagrid-classification-512p-no-gesture-150k (licencia
CC-BY-SA 4.0, misma que el HaGRID original). Como las filas estan
ordenadas por etiqueta, se localiza el rango de cada clase con una
busqueda binaria sobre la API de rows y luego se descargan las
primeras N imagenes de ese rango.

Uso:
    python -m mimic.data_curation.descargar_hagrid
"""
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

DATASET = "cj-mills/hagrid-classification-512p-no-gesture-150k"
TOTAL_FILAS = 153_735
IMAGENES_POR_CLASE = 120

# indice de la etiqueta en el ClassLabel del dataset -> carpeta destino
CLASES_OBJETIVO = {
    4: "pulgar_arriba",   # "like" en HaGRID
    9: "saludo",          # "palm" en HaGRID
}

RAIZ_DATOS = Path("data/images")


def _abrir_con_reintentos(url: str, intentos: int = 5):
    """La API publica devuelve 503 esporadicos bajo carga; se reintenta
    con espera exponencial en lugar de abortar toda la descarga."""
    peticion = Request(url, headers={"User-Agent": "mimicvision-curacion/1.0"})
    for intento in range(intentos):
        try:
            return urlopen(peticion, timeout=60)
        except Exception:
            if intento == intentos - 1:
                raise
            time.sleep(2 ** intento)


def _pedir_json(url: str) -> dict:
    with _abrir_con_reintentos(url) as respuesta:
        return json.load(respuesta)


def _etiqueta_en(offset: int) -> int:
    url = (
        f"https://datasets-server.huggingface.co/rows?dataset={quote(DATASET, safe='')}"
        f"&config=default&split=train&offset={offset}&length=1"
    )
    datos = _pedir_json(url)
    return datos["rows"][0]["row"]["label"]


def _primer_offset_de(etiqueta: int) -> int:
    """Busqueda binaria del primer offset cuya etiqueta sea >= la buscada.

    Funciona porque el dataset esta ordenado por etiqueta ascendente."""
    izquierda, derecha = 0, TOTAL_FILAS - 1
    while izquierda < derecha:
        medio = (izquierda + derecha) // 2
        if _etiqueta_en(medio) < etiqueta:
            izquierda = medio + 1
        else:
            derecha = medio
    return izquierda


def _descargar_lote(offset: int, cantidad: int, carpeta: Path, etiqueta: int) -> int:
    """Descarga un lote de filas y guarda solo las de la etiqueta pedida.

    Devuelve cuantas imagenes se guardaron."""
    url = (
        f"https://datasets-server.huggingface.co/rows?dataset={quote(DATASET, safe='')}"
        f"&config=default&split=train&offset={offset}&length={cantidad}"
    )
    datos = _pedir_json(url)
    guardadas = 0
    for fila in datos["rows"]:
        if fila["row"]["label"] != etiqueta:
            break  # se acabo el rango de la clase
        src = fila["row"]["image"]["src"]
        destino = carpeta / f"hagrid_{fila['row_idx']}.jpg"
        if not destino.exists():
            with _abrir_con_reintentos(src) as resp, open(destino, "wb") as archivo:
                archivo.write(resp.read())
        guardadas += 1
    return guardadas


def main():
    for etiqueta, nombre_clase in CLASES_OBJETIVO.items():
        carpeta = RAIZ_DATOS / nombre_clase
        carpeta.mkdir(parents=True, exist_ok=True)

        inicio = _primer_offset_de(etiqueta)
        print(f"{nombre_clase}: la clase empieza en el offset {inicio}")

        descargadas = 0
        offset = inicio
        while descargadas < IMAGENES_POR_CLASE:
            pendientes = min(100, IMAGENES_POR_CLASE - descargadas)
            guardadas = _descargar_lote(offset, pendientes, carpeta, etiqueta)
            if guardadas == 0:
                break  # el rango de la clase termino antes de lo esperado
            descargadas += guardadas
            offset += guardadas
            print(f"  {descargadas}/{IMAGENES_POR_CLASE}")
            time.sleep(1)  # cortesia con la API publica

        print(f"{nombre_clase}: {descargadas} imagenes en {carpeta}")


if __name__ == "__main__":
    main()
