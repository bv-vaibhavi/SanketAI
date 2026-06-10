import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Configuration
SIGN_CLASS = None  # Will be set by user input
SAMPLES_PER_CLASS = 30
OUTPUT_CSV = None  # Will be set based on sign class

def ensure_dataset_folder(sign_class):
    """Create dataset folder if it doesn't exist"""
    dataset_path = f"dataset/{sign_class}"
    os.makedirs(dataset_path, exist_ok=True)
    return dataset_path

def collect_data():
    """Main data collection loop"""
    global SIGN_CLASS, OUTPUT_CSV
    
    # Get sign class from user
    SIGN_CLASS = input("Enter the ISL sign class (A-Z or word name, e.g., 'A' or 'hello'): ").strip().upper()
    
    if not SIGN_CLASS:
        print("Invalid input. Exiting.")
        return
    
    # Create dataset folder and CSV file
    dataset_path = ensure_dataset_folder(SIGN_CLASS)
    OUTPUT_CSV = f"{dataset_path}/{SIGN_CLASS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"\n📹 Recording ISL sign: {SIGN_CLASS}")
    print(f"📁 Saving to: {OUTPUT_CSV}")
    print(f"📊 Target samples: {SAMPLES_PER_CLASS}")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  SPACE  -> Start/Stop recording one sample")
    print("  'c'    -> Confirm and save data")
    print("  'q'    -> Quit without saving")
    print("="*60 + "\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam!")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    collected_data = []
    is_recording = False
    current_sample = []
    sample_count = 0
    
    print(f"✅ Webcam opened. Ready to record {SIGN_CLASS}...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break
        
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        result = hands.process(rgb_frame)
        
        # Display status
        status_text = "🔴 RECORDING..." if is_recording else "⚪ IDLE (Press SPACE to record)"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 if is_recording else 100, 0), 2)
        cv2.putText(frame, f"Sign: {SIGN_CLASS} | Samples: {sample_count}/{SAMPLES_PER_CLASS}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw hand landmarks if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks if recording
                if is_recording:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    current_sample.append(landmarks)
            
            cv2.putText(frame, "✅ Hand detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "⚠️ No hand detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show instructions
        cv2.putText(frame, "Press 'c' to save | 'q' to quit", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow("ISL Data Collector", frame)
        
        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE - toggle recording
            if is_recording:
                is_recording = False
                # Save sample if it has frames
                if len(current_sample) > 5:  # Only save if we have enough frames
                    avg_landmarks = np.mean(current_sample, axis=0)
                    collected_data.append([SIGN_CLASS] + list(avg_landmarks))
                    sample_count += 1
                    print(f"✅ Sample {sample_count} recorded ({len(current_sample)} frames)")
                else:
                    print("⚠️ Sample too short. Not saved.")
                current_sample = []
            else:
                is_recording = True
                current_sample = []
                print(f"🔴 Recording sample {sample_count + 1}...")
        
        elif key == ord('c'):  # 'c' - confirm and save
            if collected_data:
                save_to_csv(collected_data)
                print(f"✅ Saved {len(collected_data)} samples to {OUTPUT_CSV}")
                break
            else:
                print("⚠️ No data to save. Keep recording!")
        
        elif key == ord('q'):  # 'q' - quit
            print("❌ Cancelled. No data saved.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def save_to_csv(data):
    """Save collected landmarks to CSV"""
    csv_headers = ['label'] + [f'x{i},y{i},z{i}'.split(',') for i in range(21)]
    csv_headers = ['label'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(data)

if __name__ == "__main__":
    print("🎯 ISL Real-Time Sign Language Translator - Data Collector")
    print("=" * 60)
    collect_data()
    print("Done!")