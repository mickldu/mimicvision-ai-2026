from ask.events import construir_event


def test_construir_event_con_los_campos_del_contrato():
    evento_preliminar = {
        "type": "brazos_cruzados", "start_time": 1.0, "end_time": 2.5, "duration_s": 1.5,
    }

    evento = construir_event(evento_preliminar, consulta="arms crossed", frame_path="evidence/e1.jpg", fuente="MIMIC+LocateAnything")

    assert evento["type"] == "brazos_cruzados"
    assert evento["start_time"] == 1.0
    assert evento["end_time"] == 2.5
    assert evento["duration_s"] == 1.5
    assert evento["query"] == "arms crossed"
    assert evento["frame_path"] == "evidence/e1.jpg"
    assert evento["source"] == "MIMIC+LocateAnything"
    assert evento["event_id"].startswith("EVT_")


def test_cada_event_id_es_distinto():
    evento_preliminar = {"type": "zen", "start_time": 0.0, "end_time": 1.0, "duration_s": 1.0}
    e1 = construir_event(evento_preliminar)
    e2 = construir_event(evento_preliminar)
    assert e1["event_id"] != e2["event_id"]
