import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import time
import pyttsx3
from threading import Thread

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load model and scaler
model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('processed/scaler.pkl')

def main_app():
    """Main UI application"""
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    sentence_buffer = []
    current_sign = ""
    sign_history = []
    prev_sign = ""
    frame_count = 0
    fps_time = time.time()
    fps = 0
    sign_hold_count = 0
    HOLD_THRESHOLD = 10
    
    print("\n" + "="*60)
    print("🎯 ISL Real-Time Sign Language Translator - Main App")
    print("="*60)
    print("\nKEYBOARD CONTROLS:")
    print("  SPACE  -> Add space between signs")
    print("  BACKSPACE -> Delete last sign")
    print("  ENTER  -> Speak the sentence")
    print("  'c'    -> Clear sentence")
    print("  'q'    -> Quit")
    print("="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        current_sign = ""
        
        # Detect hand
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Normalize
            landmarks = np.array(landmarks).reshape(21, 3)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            landmarks = landmarks.flatten().reshape(1, -1)
            
            # Scale and predict
            landmarks = scaler.transform(landmarks)
            current_sign = model.predict(landmarks)[0]
            
            # Update history
            if current_sign != prev_sign:
                sign_history.append(current_sign)
                if len(sign_history) > 5:
                    sign_history.pop(0)
                sign_hold_count = 0
            
            # Count frames sign is held
            if current_sign == prev_sign:
                sign_hold_count += 1
            
            prev_sign = current_sign
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # Hand disappeared - add sign if it was held long enough
            if prev_sign != "" and sign_hold_count >= HOLD_THRESHOLD:
                if not sentence_buffer or sentence_buffer[-1] != prev_sign:
                    sentence_buffer.append(prev_sign)
                    print(f"✅ Added '{prev_sign}' to sentence | Sentence: {''.join(sentence_buffer)}")
            
            prev_sign = ""
            current_sign = ""
            sign_hold_count = 0
        
        # FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_time
            fps = 10 / elapsed
            fps_time = time.time()
        
        # Draw UI
        cv2.rectangle(frame, (0, h-120), (w, h), (30, 30, 30), -1)
        
        # Current sign display
        cv2.putText(frame, f"Current: {current_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 2)
        
        # Sentence display
        sentence_text = ' '.join(sentence_buffer)
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # History panel
        cv2.putText(frame, "History:", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        for i, sign in enumerate(sign_history):
            cv2.putText(frame, f"  {sign}", (10, 125 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # FPS display
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "SPACE=space | BACKSPACE=del | ENTER=speak | C=clear | Q=quit", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow("ISL Sign Language Translator", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if sentence_buffer and sentence_buffer[-1] != ' ':
                sentence_buffer.append(' ')
                print(f"✅ Added space | Sentence: {''.join(sentence_buffer)}")
        
        elif key == ord('\b') or key == 8:
            if sentence_buffer:
                sentence_buffer.pop()
                print(f"🗑️  Deleted last | Sentence: {''.join(sentence_buffer)}")
        
        elif key == ord('\r') or key == 13:
            sentence_text = ''.join(sentence_buffer).strip()
            if sentence_text:
                print(f"🔊 Speaking: {sentence_text}")
                tts_engine.say(sentence_text)
                tts_engine.runAndWait()
        
        elif key == ord('c') or key == ord('C'):
            sentence_buffer = []
            print("🗑️  Cleared sentence")
        
        elif key == ord('q') or key == ord('Q'):
            print("👋 Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_app()