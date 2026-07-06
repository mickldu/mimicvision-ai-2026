"""Entrena y compara un SVM y un Random Forest sobre el vector de
features de MIMIC, y se queda con el que tenga mejor F1 macro -- no
accuracy global -- porque las clases estan desbalanceadas (HaGRID
aporta muchas mas muestras que las clases curadas a mano)."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


@dataclass
class ResultadoEntrenamiento:
    modelo: object
    nombre_modelo: str
    f1_macro_validacion: float


def entrenar_y_seleccionar(X, y) -> ResultadoEntrenamiento:
    # El SVM se envuelve en CalibratedClassifierCV porque el pipeline en
    # vivo necesita predict_proba para mostrar la confianza, y el viejo
    # SVC(probability=True) esta deprecado en sklearn 1.9 y desaparece
    # en 1.11 -- este es el reemplazo que la propia libreria recomienda.
    candidatos = {
        "svm": CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    n_muestras_clase_minima = min(np.unique(y, return_counts=True)[1])
    n_splits = max(2, min(5, n_muestras_clase_minima))
    validador = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mejor_nombre, mejor_modelo, mejor_f1 = None, None, -1.0
    for nombre, modelo in candidatos.items():
        puntajes = cross_val_score(modelo, X, y, cv=validador, scoring="f1_macro")
        f1_promedio = float(np.mean(puntajes))
        if f1_promedio > mejor_f1:
            mejor_nombre, mejor_modelo, mejor_f1 = nombre, modelo, f1_promedio

    mejor_modelo.fit(X, y)
    return ResultadoEntrenamiento(
        modelo=mejor_modelo, nombre_modelo=mejor_nombre, f1_macro_validacion=mejor_f1
    )


def guardar_modelo(modelo, ruta: Path) -> None:
    joblib.dump(modelo, ruta)


def cargar_modelo(ruta: Path):
    return joblib.load(ruta)
