import mediapipe as mp

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS  # <-- Add this line

# === Finger tip landmarks ===
FINGER_TIPS = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_TIPS[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip_id in FINGER_TIPS[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def get_gesture_label(finger_states):
    if finger_states == [1, 0, 0, 0, 1]:  
        return "Erase"
    
    total_fingers = sum(finger_states)
    return {
        1: "Writing",
        2: "Stop Writing",
        0: "Solve",
        5: "Reset"
    }.get(total_fingers, "Unknown")
