import cv2
import time

# ── Temporary mock detector (until Vaibhavi pushes her model) ──
# Once Vaibhavi pushes detector.py, replace these 3 lines with:
# from src.detector import init, update, get_current_sign

def init(): pass

def update(frame): return frame

_mock_signs = ['A', 'B', 'C', 'Hello', 'Yes', 'No', 'Help', 'Water']
_mock_index = [0]
_mock_last = [time.time()]

def get_current_sign():
    if time.time() - _mock_last[0] > 2.0:
        _mock_index[0] = (_mock_index[0] + 1) % len(_mock_signs)
        _mock_last[0] = time.time()
    return _mock_signs[_mock_index[0]]
# ── End of mock ──────────────────────────────────────────────────

# Initialize detector
init()

# Open webcam
cap = cv2.VideoCapture(0)

sentence = ""
last_sign = ""
last_sign_time = time.time()
HOLD_SECONDS = 1.5   # how long to hold a sign before it's added to sentence

print("SanketAI started!")
print("Hold a sign steady to add it to the sentence")
print("Press C to clear sentence | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Get annotated frame and current sign from detector
    frame = update(frame)
    sign = get_current_sign()

    # Sentence builder logic — only add sign if held for 1.5 seconds
    if sign and sign != last_sign:
        last_sign = sign
        last_sign_time = time.time()
    elif sign and sign == last_sign:
        held_for = time.time() - last_sign_time
        # Show a small progress bar so user knows how long they've held
        bar_width = int((held_for / HOLD_SECONDS) * 200)
        bar_width = min(bar_width, 200)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 155),
                      (0, 255, 100), -1)
        cv2.rectangle(frame, (20, 130), (220, 155), (255, 255, 255), 2)

        if held_for >= HOLD_SECONDS:
            sentence += sign + " "
            last_sign = ""   # reset so same sign can be added again

    # Display sign and sentence on screen
    cv2.putText(frame, f"Sign: {sign}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 180), 3)
    cv2.putText(frame, f"Sentence: {sentence}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "Hold 1.5s to add | C=clear | Q=quit", (20, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("SanketAI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        sentence = ""
        print("Sentence cleared")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()