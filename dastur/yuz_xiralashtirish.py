import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera ochilmadi.")
        return

    # OpenCV ichida tayyor yuz detektori bor (Haar cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi_blur = cv2.GaussianBlur(roi, (35, 35), 0)  # blur kuchi shu yerda
            frame[y:y+h, x:x+w] = roi_blur

            # Xohlasa ramka ham chizib qo'y
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Blur (press q to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
