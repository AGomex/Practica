from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2

# Tu mismo clasificador y lógica
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    countPerson = 0
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 6, minSize=(80, 80))
    for (x, y, w, h) in faces:
        countPerson += 1
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces, countPerson


def generate_frames():
    while True:
        result, video_frame = video_capture.read()
        if not result:
            break

        faces, countPerson = detect_bounding_box(video_frame)
        video_frame = cv2.flip(video_frame, 1)

        text = "Cantidad de personas: " + str(countPerson)
        cv2.putText(video_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        _, buffer = cv2.imencode('.jpg', video_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Vista de la página principal
def home(request):
    return render(request, 'deteccion/home.html')


# Vista del video (stream)
def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')