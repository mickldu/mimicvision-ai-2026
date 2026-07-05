"""Manifiesto de las 10 clases de MIMIC: de donde viene cada una y con
que etiqueta original se busca en su fuente. Ver docs/superpowers/specs/
2026-07-05-mimic-design.md seccion 2 para la justificacion completa."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DefinicionClase:
    nombre: str
    tipo: str  # "gesto" o "pose"
    fuente: str  # "hagrid" o "fotos_cc"
    etiqueta_hagrid: str | None = None  # solo si fuente == "hagrid"


CLASES = [
    DefinicionClase("saludo", "gesto", "hagrid", etiqueta_hagrid="palm"),
    DefinicionClase("pulgar_arriba", "gesto", "hagrid", etiqueta_hagrid="like"),
    DefinicionClase("mano_en_menton", "gesto", "fotos_cc"),
    DefinicionClase("manos_juntas", "gesto", "fotos_cc"),
    DefinicionClase("senalamiento", "gesto", "fotos_cc"),
    DefinicionClase("neutral", "pose", "fotos_cc"),
    DefinicionClase("zen", "pose", "fotos_cc"),
    DefinicionClase("pensando", "pose", "fotos_cc"),
    DefinicionClase("brazos_cruzados", "pose", "fotos_cc"),
    DefinicionClase("brazos_abiertos", "pose", "fotos_cc"),
]
