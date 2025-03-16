import cv2 
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0
thumb_y = 0
scroll_speed = 20  # Adjust scrolling speed

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Index finger tip
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    pyautogui.moveTo(index_x, index_y, duration=0.1)  # Smoother movement
                
                if id == 4:  # Thumb tip
                    thumb_y = screen_height / frame_height * y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    
            # Continuous scrolling based on hand movement
            if abs(index_y - thumb_y) < 70:  # Fingers close together
                if index_y < screen_height // 2 - 50:  # Move hand upwards to scroll up
                    pyautogui.scroll(scroll_speed)
                elif index_y > screen_height // 2 + 50:  # Move hand downwards to scroll down
                    pyautogui.scroll(-scroll_speed)
    
    cv2.imshow('virtual mouse', frame)
    cv2.waitKey(1)
