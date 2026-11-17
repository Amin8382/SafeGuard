from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert 'model_ready' in payload
    assert 'model_loaded' in payload


def test_classes_are_intersection_only():
    response = client.get('/classes')
    assert response.status_code == 200
    assert response.json()['classes'] == ['person', 'helmet', 'vest', 'no_helmet']


def test_model_info_exposes_thresholds():
    response = client.get('/model/info')
    assert response.status_code == 200
    payload = response.json()
    assert payload['format'] == 'ONNX'
    assert payload['confidence_threshold'] > 0
    assert payload['iou_threshold'] > 0


def test_predict_requires_exported_model():
    image = Image.new('RGB', (64, 64), 'white')
    buf = BytesIO()
    image.save(buf, format='JPEG')
    response = client.post(
        '/predict/image',
        files={'file': ('test.jpg', buf.getvalue(), 'image/jpeg')},
    )
    assert response.status_code == 503
    assert 'Model not ready' in response.json()['detail']


def test_rejects_non_image_upload():
    response = client.post(
        '/predict/image',
        files={'file': ('test.txt', b'hello', 'text/plain')},
    )
    assert response.status_code == 415
