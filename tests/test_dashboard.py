"""Dashboard document and static asset delivery checks."""


def test_dashboard_home_renders_application(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SmartHub" in response.data
    assert b"CHARGE" in response.data
    assert b"STORE" in response.data
    assert b"BORROW" in response.data


def test_dashboard_static_assets_resolve(client):
    css = client.get("/static/style.css")
    js = client.get("/static/script.js")
    assert css.status_code == 200
    assert b"focus-visible" in css.data
    assert js.status_code == 200
    assert b"/api/dashboard/summary" in js.data
    assert b"/api/charge/start" in js.data


def test_dashboard_api_data_endpoints_are_available(client):
    assert client.get("/api/dashboard/summary").status_code == 200
    assert client.get("/api/metrics/usage").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/events").status_code == 200
    assert client.get("/api/health").status_code == 200
