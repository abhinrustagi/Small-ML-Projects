# Importing the libraries
import cv2
import numpy as np

# Assigning webcam to variable cap
cap = cv2.VideoCapture(0)

# Loading the Haarcascade Face Detection Classifier
face_cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Iterator for face data storage process
skip = 0

# NumPy Array for runtime storing data of captured face
face_data = []

# Path of dataset
dataset_path = 'data/'

# Asking which person is being scanned
file_name = input("Who is being registered?\n")

# Loop to capture images
while True:
    # cap.read() returns two values: A Boolean Value stating whether the operation
    # was successful or not, and the frame information.
    ret, frame = cap.read()

    if ret == False:
        continue

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Grayscaling the Image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting any faces
    '''
        scaleFactor (1.3) =  Determines how much the image size is reduced at each image scale.
        Suppose our model inputs a 100x100 image. We capture an image of size 500x500. We will
        resize the captured image. scaleFactor 1.3 means that each step of resize, the image
        will lose 30% of its original size.
        Number of Neighbors (5) = Determines how many neighbors each candidate should have to retain it.
    '''
    # This returns the coordinates of the detected faces
    faces = face_cascade_classifier.detectMultiScale(gray_frame, 1.3, 5)

    # Sort faces on the basis of size, arranging from Largest to Smallest
    faces = sorted(faces, key = lambda f: f[2]*f[3], reverse = True)

    # Drawing Rectangles Around Detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

        # Crop out the required part : Region of Interest
        # Adding an Offset/Padding of 10 pixels
        offset = 10
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Store every 10th face
        skip += 1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

        #Displaying just the face
        cv2.imshow("Face", face_section)

    # Displaying the captured image
    cv2.imshow("Video Frame", frame)

    # Wait for user to press 'q', and then stop capturing images.
    # waitKey(1) returns a 32 bit boolean value, if we AND it with 0xFF which represents 8 1's, we
    # get an 8 bit number.
    key_pressed = cv2.waitKey(1) & 0xFF

    # ord is used to record the ASCII value of the key pressed, as an 8 bit value
    if key_pressed == ord('q'):
        break

# Convert face list to Numpy Array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Saving the data
np.save(dataset_path + file_name + '.npy', face_data)
print("Data successfully saved")

# Unbinding the Webcam
cap.release()

# Destroying any open windows
cv2.destroyAllWindows()
