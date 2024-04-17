import cv2
import math
import mediapipe as mp
import osascript




# In[8]:


class HandDetector:

    # constructor for the class
    def __init__(self, mode = False, maxHands = 1, modelC = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelC = modelC
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    # finds and draaw the hands
    def findHands(self, img):
        imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRBG)
        
        # If landmarks detected
        if self.results.multi_hand_landmarks:
            # Then draw the landmarks
            for handLandmarks in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec(color = (255, 0, 0), thickness = 2, circle_radius = 2), self.mpDraw.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2))
        return img
    
    # finds if we are losing left hand or right hand
    def findHandedness(self):
        if self.results.multi_handedness:
            handType1 = self.results.multi_handedness[0].classification[0].label
            return handType1
    
    # find the distance between 2 points
    def findDistance(self, point1, point2):
        distance = ((point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)**0.5
        return distance

        
    def findPosition(self, img):
        landmarkList = []

        # if more than one hand was detected
        if self.results.multi_hand_landmarks:

            # this line selects the first hand that was detected
            myHand = self.results.multi_hand_landmarks[0]

            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([id, cx, cy])
        return landmarkList

        
        


# In[ ]:


if __name__ == "__main__":

    # chooses the default camera
    cap = cv2.VideoCapture(0)
    #cap = cv2.flip(cap,0)

    # creates a detector object
    detector = HandDetector()


    while True:

        # --------------- reads the image captured by the camera and return 2 values
        # --------------- wether or not it was a success and the image itself
        success, img = cap.read()

        # --------------- calls the findHands function on the image
        img = detector.findHands(img)

        # --------------- call the find position and saves the list of landmark
        landmarkList = detector.findPosition(img)


        # --------------- checks wether or not we have landmarks on the hand or if there's a hand
        if len(landmarkList) != 0:


            x1, y1 = landmarkList[4][1], landmarkList[4][2]
            x2, y2 = landmarkList[8][1], landmarkList[8][2]

            # shows circles at the index and thumb tips
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 5)

            # find the distance between index and thumb
            thumbIndexDistance = detector.findDistance(landmarkList[4], landmarkList[8])

            # find the length of the index
            unitDistance = detector.findDistance(landmarkList[8], landmarkList[7])

            # the volume will be the ratio of the distance of between the index and thumb and the length of the first phalange
            volume = (thumbIndexDistance/unitDistance)

            # we want our max volumes to be between 0 and 8
            if (volume > 7):
                volume = 7
            elif(volume < 1):
                volume = 0

            # we convert the volume in percentage
            convertedVolume = (volume*100)/7

            command = "set volume output volume " + str(int(convertedVolume))
            osascript.run(command)

            # --------------- if the text detected is in the dictionary of values that we have add it to the gest_value array

            

        
        #img = cv2.resize(img, (500, 500))

        # --------------- just creates the window by which we can see the landmarks
        cv2.imshow("Volume Control With Hand Detection", img)

        #print(landmarkList)


        # --------------- allows the user to leave by pressing q
        if cv2.waitKey(1) == ord('q'):
            break


    # --------------- basically shuts down the windows and gets out of the camera    
    cv2.destroyAllWindows()
    cap.release()
