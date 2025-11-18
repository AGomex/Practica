import cv2

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
    return faces , countPerson

def Captura():
    while True:

        result, video_frame = video_capture.read()  
        if result is False:
            break  

        faces, countPerson = detect_bounding_box(
            video_frame
        ) 
        video_frame = cv2.flip(video_frame, 1)
        text = "Conteo de personas en el laboratorio: " + str(countPerson)
        cv2.putText(video_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)


        cv2.imshow(
            "Deteccion De Personas", video_frame
        )  

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()


'''
if not video_capture.isOpened():
    print("No se puede abrir la camara")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No se pudo recibir el frame")
        break
    cv2.imshow("Camara", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

'''
