import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize face mesh detector
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Track last click time to prevent multiple clicks
last_click_time = 0
click_delay = 1.0  # seconds between clicks

while True:
    # Capture and process frame
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Left eye landmarks (for right click)
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        
        # Draw left eye landmarks
        left_eye_top_x = int(left_eye_top.x * frame_w)
        left_eye_top_y = int(left_eye_top.y * frame_h)
        left_eye_bottom_x = int(left_eye_bottom.x * frame_w)
        left_eye_bottom_y = int(left_eye_bottom.y * frame_h)
        cv2.circle(frame, (left_eye_top_x, left_eye_top_y), 3, (0, 255, 255))
        cv2.circle(frame, (left_eye_bottom_x, left_eye_bottom_y), 3, (0, 255, 255))
        
        # Right eye landmarks (for left click)
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        # Draw right eye landmarks
        right_eye_top_x = int(right_eye_top.x * frame_w)
        right_eye_top_y = int(right_eye_top.y * frame_h)
        right_eye_bottom_x = int(right_eye_bottom.x * frame_w)
        right_eye_bottom_y = int(right_eye_bottom.y * frame_h)
        cv2.circle(frame, (right_eye_top_x, right_eye_top_y), 3, (255, 0, 255))
        cv2.circle(frame, (right_eye_bottom_x, right_eye_bottom_y), 3, (255, 0, 255))
        
        # Calculate eye aspect ratios
        left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y)
        
        # Display eye distances for debugging
        cv2.putText(frame, f"Left eye: {left_eye_distance:.4f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Right eye: {right_eye_distance:.4f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Check for clicks (eye blinks)
        current_time = time.time()
        if current_time - last_click_time > click_delay:
            # Left click with right eye blink
            if right_eye_distance < 0.01:
                cv2.putText(frame, "LEFT CLICK", (frame_w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pyautogui.click(button='left')
                last_click_time = current_time
            
            # Right click with left eye blink
            elif left_eye_distance < 0.01:
                cv2.putText(frame, "RIGHT CLICK", (frame_w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.click(button='right')
                last_click_time = current_time
    
    # Display the frame
    cv2.imshow('Eye-Controlled Mouse Clicks', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()