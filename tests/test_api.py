import pytest

from dga_detector.api import app


@pytest.yield_fixture
def client():
    with app.test_client() as client:
        yield client


def test_classify_valid_url(client):
    result = client.get("/get_prediction/google.com")
    assert result.status_code == 200

    # TODO: get data from schema, test the schema
    keys = result.json.keys()
    for key in ["classification", "domain", "p_dga", "p_legit"]:
        assert key in keys


def test_classify_invalid_url(client):
    result = client.get("/get_prediction/google")
    assert result.status_code == 422


@pytest.mark.parametrize(
    "url_string, expected_class",
    [
        ("google.com yahoo.com bing.com", "legit"),
        ("sflglfgldm.com lskgjflkd.org sfgskmdkfls.co.uk", "dga"),
    ],
)
def test_batch_classify_sanity_check(url_string, expected_class, client):
    data = {"url_string": url_string}
    result = client.post("/get_predictions", json=data)
    assert result.status_code == 200
    assert all([res["classification"] == expected_class for res in result.json])
    assert len(result.json) == 3


def test_batch_classify_invalid_urls(client):
    data = {"url_string": "not a url thisisntone neitheris.this"}
    result = client.post("/get_predictions", json=data)
    assert result.status_code == 422