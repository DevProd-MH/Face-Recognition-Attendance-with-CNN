import os
import cv2


def camer(srcpath, index):
    path = os.path.join(srcpath, "haarcascade/haarcascade_frontalface_default.xml")
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(path)

    # To capture video from webcam.
    cap = cv2.VideoCapture(index)

    while True:
        # Read the frame
        _, img = cap.read()
        img = cv2.flip(img, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Display
        cv2.imshow("Camera Check", img)

        # Stop if q key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
