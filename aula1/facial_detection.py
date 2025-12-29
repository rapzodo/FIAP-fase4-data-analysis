import cv2

def capture_video():
    # Inicia a captura de vídeo da webcam (0 é geralmente o índice da webcam padrão)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

capture_video()