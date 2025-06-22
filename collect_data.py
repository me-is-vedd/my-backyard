import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Prompt for label before starting webcam
current_label = input("Enter the label for this data collection session (e.g., A, B, C): ")

cap = cv2.VideoCapture(0)
collected_landmarks = []

print("Press 's' to save current frame's landmarks, 'q' to quit and export CSV.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Collect landmarks as a flat list
            flat = []
            for lm in hand_landmarks.landmark:
                flat.extend([lm.x, lm.y, lm.z])
            flat.append(current_label)
            # Show a message on the frame
            cv2.putText(frame, f"Press 's' to save for label '{current_label}'", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Hand Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and results.multi_hand_landmarks:
        collected_landmarks.append(flat)
        print(f"Saved landmarks for label '{current_label}'. Total: {len(collected_landmarks)}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ensure the data directory exists
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Save to CSV named after the label in the data directory
header = []
for i in range(21):
    header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
header.append("label")

csv_filename = os.path.join(data_dir, f"{current_label}.csv")
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(collected_landmarks)

print(f"Landmarks saved to {csv_filename}")
