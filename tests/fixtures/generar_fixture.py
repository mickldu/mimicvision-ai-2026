"""Genera una foto de prueba real (no sintetica) a partir del dataset
de ejemplo que trae scikit-image, para que los tests de landmarks
tengan una imagen de persona reproducible sin depender de internet."""
from pathlib import Path
import cv2
from skimage import data


def generar():
    imagen = data.astronaut()
    ruta = Path(__file__).parent / "persona_prueba.jpg"
    cv2.imwrite(str(ruta), cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
    return ruta


if __name__ == "__main__":
    print(generar())
