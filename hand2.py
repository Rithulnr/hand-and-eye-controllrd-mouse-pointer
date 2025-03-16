import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
smoothening = 5
prev_x, prev_y = 0, 0

click_threshold = 0.02  
scroll_threshold = 0.05  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            cur_x = np.interp(x, (0, frame.shape[1]), (0, screen_width))
            cur_y = np.interp(y, (0, frame.shape[0]), (0, screen_height))
            smooth_x = prev_x + (cur_x - prev_x) / smoothening
            smooth_y = prev_y + (cur_y - prev_y) / smoothening
            prev_x, prev_y = smooth_x, smooth_y

            pyautogui.moveTo(smooth_x, smooth_y)

            distance = np.linalg.norm([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
            if distance < click_threshold:
                pyautogui.click()

            scroll_movement = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            if abs(scroll_movement - prev_y) > scroll_threshold:
                scroll_direction = -1 if scroll_movement < prev_y else 1
                pyautogui.scroll(scroll_direction * 10)

    cv2.imshow("Hand Mouse", frame)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
