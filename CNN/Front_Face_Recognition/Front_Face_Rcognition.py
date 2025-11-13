import cv2

face_classifier = cv2.CascadeClassifier(r"C:\Users\DeLL\Python Data Science Work\Haar Cascade\haarcascade_frontalcatface.xml")

# Check if cascade file is loaded
if face_classifier.empty():
    print("Cascade failed to load!")
    exit()

image = cv2.imread(r"C:\Users\DeLL\Pictures\gettyimages-690051500-612x612.jpg")

if image is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("No faces found!")
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Faces", image)
    cv2.waitKey(0)
