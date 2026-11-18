from fastapi.testclient import TestClient
import api.main as api_module

class FakeModel:
    def predict(self, frame):
        assert "heat_exposure_index" in frame.columns
        assert "handling_damage_index" in frame.columns
        return [8.4]

def test_prediction_endpoint_executes(monkeypatch):
    monkeypatch.setattr(api_module, "_model", FakeModel())
    client = TestClient(api_module.app)
    response = client.post("/predict", json={
        "harvest_maturity_index": 85,
        "pulp_temperature_c": 24,
        "sweetness_brix": 14.5,
        "bruising_percentage": 4,
        "transport_duration_hours": 8,
        "cooling_delay_hours": 2,
        "packaging_type": "corrugated_carton"
    })
    assert response.status_code == 200
    assert response.json()["estimated_remaining_shelf_life_days"] == 8.4
