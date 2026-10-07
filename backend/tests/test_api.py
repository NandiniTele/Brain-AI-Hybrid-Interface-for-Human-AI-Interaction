from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_logs():
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_update_simulation():
    response = client.post(
        "/simulate",
        json={"dataset_name": "DEAP", "artifacts": ["Ocular (Blink)"]}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_auth_admin_login():
    response = client.post(
        "/auth/token",
        json={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["role"] == "Admin"

def test_auth_researcher_login():
    """Researcher role was previously unreachable due to None password bug — now fixed."""
    response = client.post(
        "/auth/token",
        json={"username": "researcher", "password": "research123"}
    )
    assert response.status_code == 200
    assert response.json()["role"] == "Researcher"

def test_auth_student_login():
    """Student role was previously unreachable due to None password bug — now fixed."""
    response = client.post(
        "/auth/token",
        json={"username": "student", "password": "student123"}
    )
    assert response.status_code == 200
    assert response.json()["role"] == "Student"

def test_auth_invalid_login():
    response = client.post(
        "/auth/token",
        json={"username": "wrong", "password": "wrong"}
    )
    assert response.status_code == 401

def test_train_requires_auth():
    """POST /train must reject unauthenticated requests with 401."""
    response = client.post("/train")
    assert response.status_code == 401

def test_train_accepts_valid_token(monkeypatch):
    """POST /train must accept a request with a valid JWT."""
    import main
    monkeypatch.setattr(main, "train_model", lambda **kwargs: None)
    login = client.post("/auth/token", json={"username": "admin", "password": "admin"})
    token = login.json()["access_token"]
    response = client.post(
        "/train",
        headers={"Authorization": f"Bearer {token}"}
    )
    # 200 = training started; 200 with already_training is also acceptable
    assert response.status_code == 200


