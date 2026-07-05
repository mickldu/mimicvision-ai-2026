"""Cura fotos con licencia Creative Commons para las 8 clases de MIMIC
que no existen en HaGRID, usando la API publica de Openverse.

Openverse indexa imagenes CC de Flickr, Wikimedia, rawpixel y otros, y
devuelve por cada foto su licencia, autor y atribucion -- exactamente
lo que el protocolo de datos del proyecto exige registrar.

El proceso por clase es:
  1. Buscar candidatos con varias consultas en ingles (el indice de
     Openverse responde mejor en ingles).
  2. Descargar cada candidato y verificarlo con el detector Holistic:
     si no se detecta el torso superior de una persona, se descarta.
  3. Guardar la foto redimensionada y registrar su origen y licencia
     en data/raw/cc_registro.csv para construir metadata.csv despues.

La verificacion final es humana: el script genera una hoja de contacto
por clase (data/raw/hojas_contacto/) para revisar de un vistazo que
cada foto si muestra la pose o gesto declarado, y borrar las que no.

Uso:
    python -m mimic.data_curation.descargar_fotos_cc
"""
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np

from mimic.landmarks import DetectorHolistic

RAIZ_DATOS = Path("data/images")
RUTA_REGISTRO = Path("data/raw/cc_registro.csv")
CARPETA_HOJAS = Path("data/raw/hojas_contacto")

OBJETIVO_POR_CLASE = 40
LICENCIAS = "cc0,by,by-sa"

# Consultas por clase. El solapamiento semantico entre "pensando" y
# "mano_en_menton" es real y esperado: sera parte del analisis de
# errores que pide la rubrica, no un defecto del script.
#
# Leccion de la primera pasada de curacion: la relevancia de Openverse
# cae en picada despues de la primera pagina, y las consultas genericas
# devuelven estatuas, cuadros y paisajes con gente diminuta. Por eso se
# usan muchas consultas especificas y pocas paginas por consulta.
CONSULTAS = {
    "mano_en_menton": [
        "hand on chin", "hand on chin thinking", "stroking chin",
        "thoughtful hand on chin portrait", "man stroking beard thinking",
        "woman hand on chin",
    ],
    "manos_juntas": [
        "hands clasped together person", "praying hands person", "namaste gesture",
        "person praying portrait", "woman praying", "man praying church",
        "child praying", "namaste greeting woman",
    ],
    "senalamiento": [
        "person pointing finger", "man pointing finger", "woman pointing",
        "man pointing at camera", "woman pointing finger portrait",
        "teacher pointing", "pointing at you",
    ],
    "neutral": [
        "person standing portrait", "man standing looking at camera", "woman standing portrait",
        "man full length portrait standing", "woman standing full body",
        "person standing relaxed",
    ],
    "zen": [
        "meditation pose person", "yoga meditation sitting", "zen meditation",
        "lotus position meditation", "man meditating outdoors", "woman meditating lotus",
    ],
    "pensando": [
        "pensive person portrait", "thoughtful man", "thinking woman portrait",
        "man deep in thought", "woman contemplating window", "pensive looking away",
    ],
    "brazos_cruzados": [
        "arms crossed person", "man with arms crossed", "woman arms crossed",
        "businessman arms folded", "woman arms folded portrait",
        "arms folded studio portrait", "student arms folded",
        "confident person arms folded",
    ],
    "brazos_abiertos": [
        "arms outstretched person", "open arms happy person", "arms wide open",
        "arms outstretched sunset", "man arms wide open",
        "woman celebrating arms raised", "arms spread wide person",
        "freedom arms open sky",
    ],
}

# Palabras en el titulo o las etiquetas que delatan que la imagen no es
# una foto de una persona real: estatuas, cuadros, tatuajes, carteles.
# Este filtro nacio de revisar las hojas de contacto de la primera
# pasada, donde estas categorias eran la mayor fuente de basura.
PALABRAS_EXCLUIDAS = {
    "statue", "sculpture", "monument", "painting", "drawing", "illustration",
    "engraving", "tattoo", "poster", "cartoon", "clipart", "clip art",
    "art print", "figurine", "carving", "mural", "sketch", "artwork",
    "buddha", "sign", "album", "cover",
}


def _parece_no_fotografia(resultado: dict) -> bool:
    texto = (resultado.get("title") or "").lower()
    for etiqueta in resultado.get("tags") or []:
        texto += " " + (etiqueta.get("name") or "").lower()
    return any(palabra in texto for palabra in PALABRAS_EXCLUIDAS)


def _buscar_openverse(consulta: str, pagina: int = 1) -> list[dict]:
    parametros = urlencode({
        "q": consulta,
        "license": LICENCIAS,
        "page_size": 20,
        "page": pagina,
        "filetype": "jpg",
        "mature": "false",
    })
    url = f"https://api.openverse.org/v1/images/?{parametros}"
    peticion = Request(url, headers={"User-Agent": "mimicvision-curacion/1.0"})
    try:
        with urlopen(peticion, timeout=30) as respuesta:
            return json.load(respuesta).get("results", [])
    except Exception:
        return []  # una consulta fallida no debe frenar la curacion


def _descargar_imagen(url: str) -> np.ndarray | None:
    peticion = Request(url, headers={"User-Agent": "mimicvision-curacion/1.0"})
    try:
        with urlopen(peticion, timeout=30) as respuesta:
            datos = np.frombuffer(respuesta.read(), dtype=np.uint8)
        imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if imagen is None or imagen.shape[0] < 200 or imagen.shape[1] < 200:
        return None  # demasiado pequena para landmarks confiables
    # se limita el lado mayor a 1024 px para no inflar el disco
    escala = 1024 / max(imagen.shape[:2])
    if escala < 1.0:
        imagen = cv2.resize(imagen, None, fx=escala, fy=escala)
    return imagen


def _generar_hoja_contacto(clase: str, carpeta: Path) -> None:
    """Arma una cuadricula con miniaturas numeradas de todas las fotos
    de la clase, para poder revisarlas de un solo vistazo."""
    rutas = sorted(carpeta.glob("*.jpg"))
    if not rutas:
        return
    lado = 160
    columnas = 8
    filas = (len(rutas) + columnas - 1) // columnas
    hoja = np.full((filas * lado, columnas * lado, 3), 255, dtype=np.uint8)
    for i, ruta in enumerate(rutas):
        miniatura = cv2.imread(str(ruta))
        if miniatura is None:
            continue
        miniatura = cv2.resize(miniatura, (lado, lado))
        cv2.putText(miniatura, ruta.stem.split("_")[-1], (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        fila, col = divmod(i, columnas)
        hoja[fila * lado:(fila + 1) * lado, col * lado:(col + 1) * lado] = miniatura
    CARPETA_HOJAS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CARPETA_HOJAS / f"{clase}.jpg"), hoja)


def _urls_ya_registradas() -> set:
    """Toda URL que ya paso por una corrida anterior se salta, este o no
    el archivo en disco: si fue borrada en la revision visual es porque
    no servia, y volver a bajarla desharia la curacion."""
    urls = set()
    if RUTA_REGISTRO.exists():
        with open(RUTA_REGISTRO, encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                urls.add(fila["url_origen"])
    return urls


def main():
    detector = DetectorHolistic()
    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    urls_conocidas = _urls_ya_registradas()

    registro_nuevo = not RUTA_REGISTRO.exists()
    with open(RUTA_REGISTRO, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if registro_nuevo:
            escritor.writerow([
                "archivo", "clase", "url_origen", "pagina_origen",
                "licencia", "autor", "atribucion",
            ])

        for clase, consultas in CONSULTAS.items():
            carpeta = RAIZ_DATOS / clase
            carpeta.mkdir(parents=True, exist_ok=True)
            guardadas = len(list(carpeta.glob("*.jpg")))
            ids_vistos = set()
            contador = guardadas

            for consulta in consultas:
                if guardadas >= OBJETIVO_POR_CLASE:
                    break
                for pagina in (1, 2):
                    if guardadas >= OBJETIVO_POR_CLASE:
                        break
                    for resultado in _buscar_openverse(consulta, pagina):
                        if guardadas >= OBJETIVO_POR_CLASE:
                            break
                        if resultado["id"] in ids_vistos:
                            continue
                        ids_vistos.add(resultado["id"])
                        if resultado["url"] in urls_conocidas:
                            continue
                        if _parece_no_fotografia(resultado):
                            continue

                        imagen = _descargar_imagen(resultado["url"])
                        if imagen is None:
                            continue
                        # el filtro automatico: sin torso superior visible
                        # no hay landmarks utiles, la foto no sirve
                        if detector.procesar(imagen).pose is None:
                            continue

                        # el nombre usa el id de Openverse y no un contador:
                        # un contador se reinicia entre corridas y termina
                        # sobrescribiendo fotos conservadas de pasadas previas
                        contador += 1
                        nombre = f"cc_{clase}_{resultado['id'][:8]}.jpg"
                        cv2.imwrite(str(carpeta / nombre), imagen)
                        escritor.writerow([
                            nombre, clase, resultado["url"],
                            resultado.get("foreign_landing_url", ""),
                            resultado.get("license", ""),
                            resultado.get("creator", ""),
                            resultado.get("attribution", ""),
                        ])
                        archivo.flush()
                        guardadas += 1
                    time.sleep(1)  # cortesia con la API publica

            print(f"{clase}: {guardadas} fotos")
            _generar_hoja_contacto(clase, carpeta)


if __name__ == "__main__":
    main()
