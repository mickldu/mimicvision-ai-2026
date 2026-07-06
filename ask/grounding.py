"""Cliente de LocateAnything-3B para grounding por lenguaje natural.

ADVERTENCIA DE VERIFICACION: este modulo no se pudo probar de punta a
punta en este entorno porque el modelo exige GPU con arquitectura
Ampere o superior (A100/H100) y Linux -- ver
docs/superpowers/specs/2026-07-05-ask-design.md seccion 2. El codigo de
carga e inferencia sigue el uso documentado en la model card de
nvidia/LocateAnything-3B en Hugging Face; la validacion real de esta
llamada queda pendiente para cuando se ejecute en Colab con GPU.

Las coordenadas de salida son enteros normalizados en [0, 1000]
envueltos en <box>...</box>. La documentacion no deja completamente
claro si los cuatro numeros van separados solo por comas o en
sub-etiquetas individuales, asi que parsear_cajas extrae todos los
enteros dentro de la etiqueta en vez de asumir un separador exacto --
mas robusto ante variaciones de formato que no se pudieron confirmar
sin acceso al modelo real.
"""
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODELO_LOCATEANYTHING = "nvidia/LocateAnything-3B"


class ErrorHardwareNoCompatible(RuntimeError):
    """El hardware disponible no puede correr LocateAnything-3B."""


def parsear_cajas(texto: str, ancho: int, alto: int) -> list[tuple[int, int, int, int]]:
    cajas = []
    for bloque in re.findall(r"<box>(.*?)</box>", texto, re.DOTALL):
        # Si el formato usa sub-etiquetas como <x1>500</x1>, los propios
        # nombres de etiqueta (x1, y2...) contienen digitos que
        # contaminarian la extraccion -- se quitan las etiquetas antes
        # de buscar numeros, no despues.
        bloque_sin_tags = re.sub(r"<[^>]+>", " ", bloque)
        numeros = [int(n) for n in re.findall(r"-?\d+", bloque_sin_tags)]
        if len(numeros) < 4:
            continue
        x1, y1, x2, y2 = numeros[:4]
        cajas.append((
            round(x1 / 1000 * ancho), round(y1 / 1000 * alto),
            round(x2 / 1000 * ancho), round(y2 / 1000 * alto),
        ))
    return cajas


class ClienteLocateAnything:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ErrorHardwareNoCompatible(
                "LocateAnything-3B requiere GPU CUDA (Ampere o superior). "
                "No se detecto GPU en este entorno -- correr en Colab con GPU."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODELO_LOCATEANYTHING, trust_remote_code=True
        )
        self._procesador = AutoProcessor.from_pretrained(
            MODELO_LOCATEANYTHING, trust_remote_code=True
        )
        self._modelo = AutoModel.from_pretrained(
            MODELO_LOCATEANYTHING, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda").eval()

    def localizar(self, imagen_bgr, consulta: str) -> dict:
        imagen_rgb = imagen_bgr[:, :, ::-1]
        imagen_pil = Image.fromarray(imagen_rgb)
        prompt = f"Locate all instances matching: {consulta}."

        mensajes = [{"role": "user", "content": [
            {"type": "image", "image": imagen_pil},
            {"type": "text", "text": prompt},
        ]}]
        texto = self._procesador.py_apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        imagenes, videos = self._procesador.process_vision_info(mensajes)
        entradas = self._procesador(
            text=[texto], images=imagenes, videos=videos, return_tensors="pt"
        ).to("cuda")

        respuesta = self._modelo.generate(
            pixel_values=entradas["pixel_values"].to(torch.bfloat16),
            input_ids=entradas["input_ids"],
            attention_mask=entradas["attention_mask"],
            image_grid_hws=entradas.get("image_grid_hws"),
            tokenizer=self._tokenizer,
            max_new_tokens=512,
            generation_mode="hybrid",
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

        alto, ancho = imagen_bgr.shape[:2]
        cajas = parsear_cajas(respuesta, ancho, alto)
        return {"consulta": consulta, "cajas": cajas, "texto_crudo": respuesta}
