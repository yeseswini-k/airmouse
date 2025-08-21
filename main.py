import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothening = 7
clicking = False

def get_pos(lms, shape, idx):
    h, w, _ = shape
    return int(lms.landmark[idx].x * w), int(lms.landmark[idx].y * h)

def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

        index = get_pos(lm, img.shape, 8)
        thumb = get_pos(lm, img.shape, 4)
        middle = get_pos(lm, img.shape, 12)

        cv2.circle(img, index, 8, (255, 0, 255), cv2.FILLED)

        # Cursor movement
        x = np.interp(index[0], (0, 640), (0, screen_w))
        y = np.interp(index[1], (0, 480), (0, screen_h))
        curr_x = prev_x + (x - prev_x) / smoothening
        curr_y = prev_y + (y - prev_y) / smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Left click
        if dist(index, thumb) < 40:
            if not clicking:
                pyautogui.click()
                clicking = True
                cv2.putText(img, "Left Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            clicking = False

        # Right click
        if dist(middle, thumb) < 40:
            pyautogui.rightClick()
            cv2.putText(img, "Right Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
