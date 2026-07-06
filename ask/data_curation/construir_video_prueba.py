"""Descarga 5 clips de video con licencia libre de Pexels y los
concatena en un unico video de prueba con limites de tiempo conocidos
-- la verdad de terreno del timeline sin necesidad de anotacion
manual. Cada clip se ajusta a un lienzo comun con letterbox (no con un
recorte 'cover'): un recorte cover llego a cortar la cara en un clip
vertical de prueba, asi que se prefiere perder algo de area con barras
negras antes que perder contenido del cuerpo que MediaPipe necesita.

Uso:
    python -m ask.data_curation.construir_video_prueba
"""
import csv
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

CARPETA_VIDEOS = Path("data/videos")
RUTA_VIDEO_PRUEBA = CARPETA_VIDEOS / "video_prueba.mp4"
RUTA_REGISTRO = CARPETA_VIDEOS / "video_prueba_registro.csv"

ANCHO_CANVAS, ALTO_CANVAS = 640, 360
FPS_SALIDA = 25.0

LICENCIA_PEXELS = "Pexels License (libre para uso personal y comercial)"

SEGMENTOS = [
    {
        "nombre": "charla", "clase_esperada": "neutral",
        "url": "https://videos.pexels.com/video-files/4106314/4106314-hd_1280_720_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/a-man-talking-with-hand-gestures-4106314/",
    },
    {
        "nombre": "pulgar", "clase_esperada": "pulgar_arriba",
        "url": "https://videos.pexels.com/video-files/8627026/8627026-hd_1920_1080_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/man-giving-thumbs-up-8627026/",
    },
    {
        "nombre": "cruzados", "clase_esperada": "brazos_cruzados",
        "url": "https://videos.pexels.com/video-files/7686684/7686684-hd_1080_1920_24fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/people-crossed-arms-and-posing-7686684/",
    },
    {
        "nombre": "saludo", "clase_esperada": "saludo",
        "url": "https://videos.pexels.com/video-files/4586958/4586958-uhd_3840_2160_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/a-man-greeting-and-waving-his-hand-4586958/",
    },
    {
        "nombre": "senalar", "clase_esperada": "senalamiento",
        "url": "https://videos.pexels.com/video-files/6974223/6974223-uhd_2160_3840_25fps.mp4",
        "pagina_origen": "https://www.pexels.com/video/man-pointing-at-the-camera-6974223/",
    },
]


def ajustar_a_canvas(frame: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Redimensiona un frame para que quepa completo dentro de un
    lienzo fijo (letterbox), sin recortar contenido."""
    h, w = frame.shape[:2]
    escala = min(ancho / w, alto / h)
    nuevo_w, nuevo_h = int(w * escala), int(h * escala)
    redimensionado = cv2.resize(frame, (nuevo_w, nuevo_h))
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    x0 = (ancho - nuevo_w) // 2
    y0 = (alto - nuevo_h) // 2
    lienzo[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = redimensionado
    return lienzo


def _descargar(url: str, destino: Path, intentos: int = 5) -> None:
    if destino.exists():
        return
    peticion = Request(url, headers={"User-Agent": "mimicvision-curacion/1.0"})
    for intento in range(intentos):
        try:
            with urlopen(peticion, timeout=90) as respuesta, open(destino, "wb") as archivo:
                archivo.write(respuesta.read())
            return
        except Exception:
            if intento == intentos - 1:
                raise
            time.sleep(10 * (2 ** intento))


def main():
    CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(
        str(RUTA_VIDEO_PRUEBA), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_SALIDA, (ANCHO_CANVAS, ALTO_CANVAS),
    )

    filas_registro = []
    tiempo_acumulado = 0.0

    for segmento in SEGMENTOS:
        ruta_local = CARPETA_VIDEOS / f"candidato_{segmento['nombre']}.mp4"
        _descargar(segmento["url"], ruta_local)

        captura = cv2.VideoCapture(str(ruta_local))
        inicio = tiempo_acumulado
        n_frames_escritos = 0

        while True:
            hay_frame, frame = captura.read()
            if not hay_frame:
                break
            escritor.write(ajustar_a_canvas(frame, ANCHO_CANVAS, ALTO_CANVAS))
            n_frames_escritos += 1
        captura.release()

        duracion = n_frames_escritos / FPS_SALIDA
        tiempo_acumulado += duracion
        filas_registro.append({
            "segmento": segmento["nombre"],
            "clase_esperada": segmento["clase_esperada"],
            "inicio_s": round(inicio, 2),
            "fin_s": round(tiempo_acumulado, 2),
            "licencia": LICENCIA_PEXELS,
            "pagina_origen": segmento["pagina_origen"],
        })
        print(f"{segmento['nombre']}: {inicio:.1f}s - {tiempo_acumulado:.1f}s")

    escritor.release()

    with open(RUTA_REGISTRO, "w", newline="", encoding="utf-8") as archivo:
        escritor_csv = csv.DictWriter(archivo, fieldnames=list(filas_registro[0].keys()))
        escritor_csv.writeheader()
        escritor_csv.writerows(filas_registro)

    print(f"Video de prueba: {RUTA_VIDEO_PRUEBA} ({tiempo_acumulado:.1f}s totales)")


if __name__ == "__main__":
    main()
