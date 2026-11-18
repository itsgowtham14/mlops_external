import pytest
from pydantic import ValidationError
from api.main import ShipmentInput

def valid_payload():
    return {
        "harvest_maturity_index": 85,
        "pulp_temperature_c": 24,
        "sweetness_brix": 14.5,
        "bruising_percentage": 4,
        "transport_duration_hours": 8,
        "cooling_delay_hours": 2,
        "packaging_type": "corrugated_carton"
    }

def test_valid_input_is_accepted():
    assert ShipmentInput(**valid_payload()).harvest_maturity_index == 85

@pytest.mark.parametrize("field,value", [
    ("harvest_maturity_index", 120),
    ("pulp_temperature_c", 60),
    ("sweetness_brix", 35),
    ("bruising_percentage", 120),
    ("transport_duration_hours", 250),
    ("cooling_delay_hours", 90),
    ("packaging_type", "wooden_box")
])
def test_invalid_input_is_rejected(field, value):
    payload = valid_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        ShipmentInput(**payload)
