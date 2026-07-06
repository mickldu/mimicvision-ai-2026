from ask.store import cargar_eventos, guardar_eventos


def test_guardar_y_cargar_eventos_hace_round_trip(tmp_path):
    ruta = tmp_path / "event_store.json"
    eventos = [{"event_id": "EVT_0001", "type": "saludo", "start_time": 0.0}]

    guardar_eventos(eventos, str(ruta))
    cargados = cargar_eventos(str(ruta))

    assert cargados == eventos
