import cv2
import autopy
import mediapipe as mp
import time

### VARIABLES
# set previous time
pTime = 0

# get screen size
width, height = autopy.screen.size()

# get video source
cap = cv2.VideoCapture(1)

# settings and var
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_tracking_confidence=0.5, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# взял размер изображения чтобы не делать это каждый фрейм, поскольку разрешение не может быть динамическим как бы
_, img = cap.read()
# get size of video
h, w, _ = img.shape

### START CYCLE
while True:
    _, img = cap.read()

    # get result of work
    result = hands.process(img)

    # if result is none
    if result.multi_hand_landmarks:

        # lm is coordinates
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            # перевод координат из видео в координаты плоскости экрана
            cursorX, cursorY = int(lm.x *w), int(lm.y*h)

            if id ==8:
                cv2.circle(img, (cursorX, cursorY), 5, (255,255,255))
                autopy.mouse.move(cursorX*width / w, cursorY*height /h)

        #mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
    ### FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    ### RENDERING
    cv2.imshow("tracking hands", img)
    cv2.waitKey(1)