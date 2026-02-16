import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera ochilmadi. VideoCapture(0) o'rniga 1 ham sinab ko'r.")
        return

    # Fonni ajratib beruvchi tayyor algoritm (juda qulay)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=25, detectShadows=True
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Mask: harakat bo'lgan joylar oq (255) bo'ladi
        fgmask = backsub.apply(frame)

        # Soyalarni (kulrang) kamaytirish uchun threshold
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Shovqinni tozalash (nuqta-nuqta bo'lib ketmasin)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # Konturlarni topamiz
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1200:  # kichik harakatlarni e'tiborsiz qoldirish
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Motion Box (press q to exit)", frame)
        cv2.imshow("Mask", fgmask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

