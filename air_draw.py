
import mediapipe as mp

def are_all_fingers_up(hand_landmarks):
    """
    Determine if all fingers are extended based on the hand landmarks.
    Returns a tuple containing a boolean value indicating whether all fingers are up,
    and a list containing True or False for each finger indicating whether it is up or down.
    """
    fingertip_indices = [mp.solutions.hands.HandLandmark.THUMB_TIP,
                         mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                         mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                         mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                         mp.solutions.hands.HandLandmark.PINKY_TIP]

    lower_joint_indices = [mp.solutions.hands.HandLandmark.THUMB_IP,
                           mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                           mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                           mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                           mp.solutions.hands.HandLandmark.PINKY_PIP]

    finger_status = [False] * len(fingertip_indices)  # Initialize all fingers as down

    # Check index finger separately
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_lower_joint = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    if index_finger_tip.y < index_finger_lower_joint.y:
        finger_status[1] = True  # Index finger is up

    for i, (fingertip, lower_joint) in enumerate(zip(fingertip_indices[2:], lower_joint_indices[2:]), start=2):
        fingertip_pos = hand_landmarks.landmark[fingertip]
        lower_joint_pos = hand_landmarks.landmark[lower_joint]

        # Check if the fingertip is above the lower joint (y-coordinate comparison)
        finger_up = fingertip_pos.y < lower_joint_pos.y

        # If fingertip is above the lower joint, mark the finger as up
        if finger_up:
            finger_status[i] = True

    # Check if all fingers are up or not
    if finger_status[1] and finger_status[2] and finger_status[3] and finger_status[4] == True:
        return True, finger_status
    else:
        return False, finger_status

    # return all_fingers_up, finger_status

