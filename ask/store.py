"""Persistencia de eventos en un archivo JSON simple -- un video o una
sesion en vivo generan decenas de eventos, no cientos de miles, asi que
SQLite no aporta nada a esta escala (mismo criterio de simplicidad que
llevo a usar similitud coseno en vez de FAISS en MATCH)."""
import json
from pathlib import Path


def guardar_eventos(eventos: list[dict], ruta: str = "data/event_store.json") -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(eventos, archivo, indent=2, ensure_ascii=False)


def cargar_eventos(ruta: str = "data/event_store.json") -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)
