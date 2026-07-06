import numpy as np

from ask.data_curation.construir_video_prueba import ajustar_a_canvas


def test_ajustar_a_canvas_produce_el_tamano_exacto_pedido():
    frame_vertical = np.zeros((1920, 1080, 3), dtype=np.uint8)
    resultado = ajustar_a_canvas(frame_vertical, ancho=640, alto=360)
    assert resultado.shape == (360, 640, 3)


def test_ajustar_a_canvas_no_recorta_contenido_de_un_frame_vertical():
    # Un bloque blanco en el centro del frame original (no un solo pixel,
    # que se diluiria con el reescalado) debe seguir presente despues del
    # ajuste, en vez de quedar recortado como pasaria con un 'cover crop'
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    frame[860:1060, 440:640] = 255  # bloque de 200x200 en el centro

    resultado = ajustar_a_canvas(frame, ancho=640, alto=360)

    columna_central = resultado[:, 320]  # columna central del canvas de salida
    assert columna_central.max() == 255  # el contenido blanco sigue presente
