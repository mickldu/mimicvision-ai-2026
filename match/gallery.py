"""Carga la galeria y las consultas de evaluacion de MATCH a partir del
mismo data/metadata.csv que ya arma MIMIC -- no se cura un dataset
nuevo, se reutiliza el existente con su split fijo (train+val como
galeria, test como consultas, sin fuga de datos entre ambos)."""
import pandas as pd


def cargar_galeria_y_consultas(ruta_metadata: str = "data/metadata.csv"):
    metadata = pd.read_csv(ruta_metadata)
    galeria = metadata[metadata["split"].isin(["train", "val"])].reset_index(drop=True)
    consultas = metadata[metadata["split"] == "test"].reset_index(drop=True)
    return galeria, consultas
