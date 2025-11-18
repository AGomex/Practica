import cv2
import numpy as np
from unittest.mock import patch
from Deteccion.views import detect_bounding_box, generate_frames
from django.test import TestCase, Client
from django.urls import reverse
from django.http import StreamingHttpResponse

def fake_image(width=640, height=480, with_face=False):
    """
    Crea una imagen negra de prueba.
    Si with_face=True dibuja un rectángulo simulando un rostro.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if with_face:
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    return img


# ===============================================================
#  UNIT TESTS: detect_bounding_box
# ===============================================================

def test_detect_bounding_box_no_faces():
    img = fake_image()
    with patch.object(cv2.CascadeClassifier, "detectMultiScale", return_value=[]):
        faces, count = detect_bounding_box(img)
    assert isinstance(faces, list)
    assert faces == []
    assert count == 0


def test_detect_bounding_box_with_faces():
    img = fake_image()
    fake_face = [(10, 20, 100, 100)]
    with patch.object(cv2.CascadeClassifier, "detectMultiScale", return_value=fake_face):
        faces, count = detect_bounding_box(img)
    assert isinstance(faces, list)
    assert faces == fake_face
    assert count == 1


# ===============================================================
#  UNIT TESTS: generate_frames
# ===============================================================

@patch("Deteccion.views.video_capture")
def test_generate_frames_single_frame(mock_capture):
    fake_frame = fake_image()
    mock_capture.read.side_effect = [(True, fake_frame), (False, None)]
    generator = generate_frames()
    frame = next(generator)
    assert b"--frame" in frame
    assert b"Content-Type: image/jpeg" in frame


@patch("Deteccion.views.video_capture")
def test_generate_frames_yields_multiple_frames(mock_capture):
    fake_frame1 = fake_image()
    fake_frame2 = fake_image()
    mock_capture.read.side_effect = [(True, fake_frame1), (True, fake_frame2), (False, None)]
    generator = generate_frames()
    frame1 = next(generator)
    frame2 = next(generator)
    assert frame1 != frame2
    assert b"--frame" in frame1
    assert b"--frame" in frame2



# ===============================================================
#  INTEGRATION TESTS: vistas Django
# ===============================================================

class ViewsIntegrationTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_view_loads_correct_template(self):
        """Verifica que la vista 'home' carga correctamente la plantilla."""
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "deteccion/home.html")

    @patch("Deteccion.views.generate_frames")
    def test_video_feed_view_streams_data(self, mock_gen):
        """Verifica StreamingHttpResponse."""
        mock_gen.return_value = iter([b"frame"])
        response = self.client.get(reverse("video_feed"))
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response, StreamingHttpResponse)
        self.assertEqual(response["Content-Type"], "multipart/x-mixed-replace; boundary=frame")

    @patch("Deteccion.views.generate_frames")
    def test_video_feed_streams_multiple_frames(self, mock_gen):
        """Verifica que el stream devuelve múltiples frames correctamente."""
        mock_gen.return_value = iter([b"frame1", b"frame2", b"frame3"])
        response = self.client.get(reverse("video_feed"))
        streamed = list(response.streaming_content)
        self.assertGreaterEqual(len(streamed), 3)
        self.assertIn(b"frame1", streamed)
        self.assertIn(b"frame2", streamed)
        self.assertIn(b"frame3", streamed)


# ===============================================================
#  FUNCTIONAL TESTS: flujo de la aplicación
# ===============================================================

class FunctionalUITest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_loads(self):
        """Valida que la página de inicio se carga correctamente."""
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "deteccion/home.html")


class FunctionalVideoTest(TestCase):
    @patch("Deteccion.views.video_capture")
    def test_video_feed_streams_frames(self, mock_capture):
        """Verifica que el stream de video devuelve frames correctamente."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture.read.side_effect = [(True, fake_frame), (True, fake_frame), (False, None)]
        generator = generate_frames()
        frames = [next(generator), next(generator)]
        for frame in frames:
            self.assertIn(b"--frame", frame)
            self.assertIn(b"Content-Type: image/jpeg", frame)


class FunctionalFlowTest(TestCase):
    @patch("Deteccion.views.generate_frames")
    def test_access_home_and_video_feed(self, mock_gen):
        """Valida flujo completo de la app (home - video_feed)."""
        mock_gen.return_value = iter([b"frame1", b"frame2"])
        # Home
        home_resp = self.client.get(reverse("home"))
        self.assertEqual(home_resp.status_code, 200)
        # Video feed
        video_resp = self.client.get(reverse("video_feed"))
        self.assertIsInstance(video_resp, StreamingHttpResponse)
        self.assertEqual(video_resp["Content-Type"], "multipart/x-mixed-replace; boundary=frame")
