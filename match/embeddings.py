"""Embeddings visuales modernos (ruta C) con SigLIP2.

Nota de API verificada durante el desarrollo: AutoModel.get_image_features()
en esta version de transformers no devuelve un tensor plano, sino un
BaseModelOutputWithPooling. El vector que se usa como embedding de la
imagen es su atributo pooler_output.

Se eligio SigLIP2 (Apache-2.0, sin gate) en vez de DINOv3 porque los
modelos facebook/dinov3-* estan gated en Hugging Face y exigirian que
cada persona que corra el notebook solicite acceso y configure un
token -- rompe la portabilidad entre local y Colab.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODELO_SIGLIP2 = "google/siglip2-base-patch16-224"


class ExtractorSigLIP2:
    def __init__(self):
        self._procesador = AutoProcessor.from_pretrained(MODELO_SIGLIP2)
        self._modelo = AutoModel.from_pretrained(MODELO_SIGLIP2)
        self._modelo.eval()

    def extraer(self, imagen_bgr: np.ndarray) -> np.ndarray:
        imagen_rgb = imagen_bgr[:, :, ::-1]
        imagen_pil = Image.fromarray(imagen_rgb)
        entradas = self._procesador(images=imagen_pil, return_tensors="pt")
        with torch.no_grad():
            salida = self._modelo.get_image_features(**entradas)
        return salida.pooler_output[0].numpy()
