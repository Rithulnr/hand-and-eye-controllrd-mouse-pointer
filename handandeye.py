import cv2
import mediapipe as mp
import pyautogui
import time

cam = cv2.VideoCapture(0)


face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

last_click_time = 0
click_delay = 1.0  
blink_threshold = 0.015 

index_y = 0
thumb_y = 0
scroll_speed = 5  

while True:
   
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
   
    face_output = face_mesh.process(rgb_frame)
    

    if face_output.multi_face_landmarks:
        for face_landmarks in face_output.multi_face_landmarks:
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            
            left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y)
            
            current_time = time.time()
            if current_time - last_click_time > click_delay:
                if right_eye_distance < blink_threshold:
                    pyautogui.click(button='left')
                    last_click_time = current_time
                elif left_eye_distance < blink_threshold:
                    pyautogui.click(button='right')
                    last_click_time = current_time
    
    hand_output = hands_detector.process(rgb_frame)
    hands = hand_output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                
                if id == 8: 
                    index_x = screen_w / frame_w * x
                    index_y = screen_h / frame_h * y
                    pyautogui.moveTo(index_x, index_y, duration=0.1)
                
                if id == 4: 
                    thumb_y = screen_h / frame_h * y
            
          
            if index_y and thumb_y:
                if abs(index_y - thumb_y) < 70:
                    if index_y < screen_h // 2 - 50:
                        pyautogui.scroll(scroll_speed)
                    elif index_y > screen_h // 2 + 50:
                        pyautogui.scroll(-scroll_speed)
    
  
    cv2.imshow('Hand & Eye Gesture Controlled Mouse', frame)
    
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
cam.release()
cv2.destroyAllWindows()
