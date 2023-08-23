from cvzone.HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import speech_recognition as sr
# Parameters
width, height = 1280, 520
gestureThreshold = 700
folderPath = "Presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

recognizer = sr.Recognizer()
# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 30
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Zoom Parameters
zoomScale = 1.0
zoomSpeed = 0.02

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        # Zoom Gesture
        if fingers == [0, 1, 1, 1, 1]:
            zoomScale += zoomSpeed
            print("Zoom In:", zoomScale)
        if fingers == [1, 1, 1, 1, 1]:
            zoomScale -= zoomSpeed
            if zoomScale < 1.0:
                zoomScale = 1.0
            print("Zoom Out:", zoomScale)
            #go to slide feature
        if fingers == [1, 0, 0, 0, 0]:
            buttonPressed = True
            print("Go to specific slide gesture detected")
            try:
                # Prompt user for slide number using voice command
                with sr.Microphone() as source:
                    print("Listening for slide number...")
                    audio = recognizer.listen(source)
                # Recognize the voice command
                command = recognizer.recognize_google(audio).lower()
                print("Voice Command:", command)
                # Extract slide number from command
                slide_number = int(command.split("slide")[-1].strip())
                # Verify and navigate to the desired slide
                if 0 <= slide_number < len(pathImages):
                    imgNumber = slide_number
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                    print(f"Going to slide {slide_number}")
                else:
                    print("Invalid slide number")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    # ... (rest of your code)






    else:
        annotationStart = False

    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    imgCurrent = cv2.resize(imgCurrent, None, fx=zoomScale, fy=zoomScale)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    cv2.imshow("Slides", imgCurrent)
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
