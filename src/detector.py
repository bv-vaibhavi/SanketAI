import cv2
import mediapipe as mp
import numpy as np
import joblib
import json

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load trained model
model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('processed/scaler.pkl')

# Load class mapping
with open('processed/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

# Reverse mapping: index -> label
idx_to_label = {int(k): v for k, v in class_mapping.items()}

# Global state for debouncing
current_sign = ""
sign_buffer = []
DEBOUNCE_FRAMES = 5

def predict_sign(landmarks):
    """Predict ISL sign from hand landmarks"""
    global current_sign, sign_buffer
    
    # Normalize: subtract wrist point
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    landmarks = landmarks.flatten().reshape(1, -1)
    
    # Scale
    landmarks = scaler.transform(landmarks)
    
    # Predict (returns string label directly)
    pred_label = model.predict(landmarks)[0]
    
    return pred_label

def get_current_sign():
    """Public interface: returns current detected sign"""
    return current_sign

def run_detector():
    """Main real-time detection loop"""
    global current_sign, sign_buffer
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened. Running real-time detection...")
    print("Press 'q' to quit\n")
    
    fps_clock = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        # Detect hand
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Predict
            pred_sign = predict_sign(landmarks)
            
            # Debounce: only change if same sign appears 5 frames in a row
            if len(sign_buffer) == 0 or sign_buffer[-1] == pred_sign:
                sign_buffer.append(pred_sign)
                if len(sign_buffer) > DEBOUNCE_FRAMES:
                    sign_buffer.pop(0)
            else:
                sign_buffer = [pred_sign]
            
            if len(sign_buffer) == DEBOUNCE_FRAMES:
                current_sign = sign_buffer[0]
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display prediction
            cv2.putText(frame, f"Sign: {current_sign}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {len(sign_buffer)}/{DEBOUNCE_FRAMES}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            current_sign = ""
            sign_buffer = []
            cv2.putText(frame, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display
        cv2.imshow("ISL Real-Time Detector", frame)
        
        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎯 ISL Real-Time Sign Language Detector")
    print("=" * 60)
    run_detector()