"""Une deteccion de landmarks, features y clasificacion en un solo
resultado por frame (PerceptionResult), tal como lo definimos en el
diseno transversal del proyecto."""
import numpy as np

from mimic.features import NARIZ, construir_vector_features


def procesar_frame(frame, detector, modelo, timestamp: float) -> dict:
    resultado_landmarks = detector.procesar(frame)

    if resultado_landmarks.pose is None:
        return {
            "frame": frame,
            "bbox_persona": None,
            "landmarks_normalizados": None,
            "etiqueta_pose": None,
            "confianza": 0.0,
            "timestamp": timestamp,
        }

    # Si el rostro no fue detectado (persona de espaldas o muy lejos),
    # la nariz de la pose sirve como sustituto razonable del menton:
    # ambos estan en la cabeza y la feature mano-menton sigue teniendo
    # sentido como "mano cerca de la cara".
    menton = resultado_landmarks.menton
    if menton is None:
        menton = resultado_landmarks.pose[NARIZ]

    vector = construir_vector_features(resultado_landmarks.pose, menton)
    vector = np.array(vector).reshape(1, -1)

    etiqueta = modelo.predict(vector)[0]
    probabilidades = modelo.predict_proba(vector)[0]
    indice_clase = list(modelo.classes_).index(etiqueta)
    confianza = float(probabilidades[indice_clase])

    return {
        "frame": frame,
        "bbox_persona": None,
        "landmarks_normalizados": vector.tolist()[0],
        "etiqueta_pose": etiqueta,
        "confianza": confianza,
        "timestamp": timestamp,
    }
