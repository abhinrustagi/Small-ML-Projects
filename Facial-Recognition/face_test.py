# Importing the libraries
import cv2
import numpy as np
import os

# Calculate distance between two points
def distance(v1, v2):
    # Euclidean distance
    return np.sqrt(((v1-v2)**2).sum())

# KNN Implementation
def KNN(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        #Compute distance from test point
        d = distance(test, ix)
        dist.append([d, iy])

    # Sort based on distance and get top k
    dk = sorted(dist, key = lambda x: x[0])[:k]

    # Retrieve only the labels
    labels = np.array(dk)[:,-1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts = True)

    # Find max frequency and corresponding label
    index = np.argmax(output[1])

    # Return the nearest label
    return output[0][index]


# Assigning webcam to variable cap
cap = cv2.VideoCapture(0)

# Loading the Haarcascade Face Detection Classifier
face_cascade_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Iterator for face data storage process
skip = 0

# NumPy Array for runtime storing data of captured face
face_data = []

labels = []

class_id = 0

# Dictionary of IDs - Names
names = {}

# Path of dataset
dataset_path = 'data/'

# Data Loading/Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):

        # Create a mapping between class ID and labels
        names[class_id] = fx[:-4]

        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

    # Create labels
    target = class_id*np.ones((data_item.shape[0],))
    class_id += 1
    labels.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis = 1)

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

    # Drawing Rectangles Around Detected faces
    for (x, y, w, h) in faces:
        # Crop out the required part : Region of Interest
        # Adding an Offset/Padding of 10 pixels
        offset = 10
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Predicting
        out = KNN(trainset, face_section.flatten())

        # Displaying Name on the screen and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)

    # Displaying
    cv2.imshow("Video Frame", frame)

    # Wait for user to press 'q', and then stop capturing images.
    # waitKey(1) returns a 32 bit boolean value, if we AND it with 0xFF which represents 8 1's, we
    # get an 8 bit number.
    key_pressed = cv2.waitKey(1) & 0xFF

    # ord is used to record the ASCII value of the key pressed, as an 8 bit value
    if key_pressed == ord('q'):
        break

# Unbinding the Webcam
cap.release()

# Destroying any open windows
cv2.destroyAllWindows()
