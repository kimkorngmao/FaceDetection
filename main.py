import cv2 as cv

cap = cv.VideoCapture(0)

# Load the open-source face detection model
face_model = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_model.detectMultiScale(gray_frame)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame with face detection
    cv.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
