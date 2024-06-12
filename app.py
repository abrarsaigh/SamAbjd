from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import os
from air_draw import are_all_fingers_up
from main1 import ModelHandler
import landmarks as htm
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


# Configuring the database
# Using SQLite for simplicity; in a production app, consider PostgreSQL, MySQL, etc.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # The 'name' column
    file_path = db.Column(db.String(255), unique=True, nullable=False)
    description = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f"<Video {self.name}>"

# Create the database and tables
if __name__ == '__main__':
    with app.app_context():  # Ensure all operations are within the context
        db.drop_all()  # Optional: Drop existing tables
        db.create_all()  # Create new tables

        # Define video samples
        video_samples = [
            {'name': 'alef', 'file_path': '/static/letterVideos/1.mp4', 'description': 'Letter Alef'},
            {'name': 'ba', 'file_path': '/static/letterVideos/2.mp4', 'description': 'Letter Ba'},
            {'name': 'ta', 'file_path': '/static/letterVideos/3.mp4', 'description': 'Letter Ta'},
            {'name': 'tha', 'file_path': '/static/letterVideos/4.mp4', 'description': 'Letter Tha'},
            {'name': 'jeem', 'file_path': '/static/letterVideos/5.mp4', 'description': 'Letter Jeem'},
            {'name': 'ha', 'file_path': '/static/letterVideos/6.mp4', 'description': 'Letter Ha'},
            {'name': 'kha', 'file_path': '/static/letterVideos/7.mp4', 'description': 'Letter Kha'},
            {'name': 'dal', 'file_path': '/static/letterVideos/8.mp4', 'description': 'Letter Dal'},
            {'name': 'thal', 'file_path': '/static/letterVideos/9.mp4', 'description': 'Letter Thal'},
            {'name': 'ra', 'file_path': '/static/letterVideos/10.mp4', 'description': 'Letter Ra'},
            {'name': 'zain', 'file_path': '/static/letterVideos/11.mp4', 'description': 'Letter Zain'},
            {'name': 'seen', 'file_path': '/static/letterVideos/12.mp4', 'description': 'Letter Seen'},
            {'name': 'sheen', 'file_path': '/static/letterVideos/13.mp4', 'description': 'Letter Sheen'},
            {'name': 'saad', 'file_path': '/static/letterVideos/14.mp4', 'description': 'Letter Saad'},
            {'name': 'daad', 'file_path': '/static/letterVideos/15.mp4', 'description': 'Letter Daad'},
            {'name': 'ta', 'file_path': '/static/letterVideos/16.mp4', 'description': 'Letter Ta'},
            {'name': 'za', 'file_path': '/static/letterVideos/17.mp4', 'description': 'Letter Za'},
            {'name': 'ain', 'file_path': '/static/letterVideos/18.mp4', 'description': 'Letter Ain'},
            {'name': 'ghain', 'file_path': '/static/letterVideos/19.mp4', 'description': 'Letter Ghain'},
            {'name': 'fa', 'file_path': '/static/letterVideos/20.mp4', 'description': 'Letter Fa'},
            {'name': 'qaaf', 'file_path': '/static/letterVideos/21.mp4', 'description': 'Letter Qaaf'},
            {'name': 'kaaf', 'file_path': '/static/letterVideos/22.mp4', 'description': 'Letter Kaaf'},
            {'name': 'laam', 'file_path': '/static/letterVideos/23.mp4', 'description': 'Letter Laam'},
            {'name': 'meem', 'file_path': '/static/letterVideos/24.mp4', 'description': 'Letter Meem'},
            {'name': 'noon', 'file_path': '/static/letterVideos/25.mp4', 'description': 'Letter Noon'},
            {'name': 'haa', 'file_path': '/static/letterVideos/26.mp4', 'description': 'Letter Haa'},
            {'name': 'waaw', 'file_path': '/static/letterVideos/27.mp4', 'description': 'Letter Waaw'},
            {'name': 'yaa', 'file_path': '/static/letterVideos/28.mp4', 'description': 'Letter Yaa'},
        ]

        # Loop through the video samples and add each to the database
        for sample in video_samples:
            video = Video(
                name=sample['name'],
                file_path=sample['file_path'],
                description=sample['description']
            )
            db.session.add(video)

        db.session.commit()  # Commit changes to the database


model_path = "CNNmodelDropOUT.h5"
model_handler = ModelHandler(model_path)  # Ensure this class is ready for prediction

is_drawing_enabled = True
global Previous_Recognized_letter
Previous_Recognized_letter = False
prev_index_finger_tip = None

global isCorrectFlag
global isNOTCorrectFlag
isCorrectFlag = False
isNOTCorrectFlag = False

global draw_color
draw_color = (255, 255, 255)  # Default to white

global target_value
target_value = 1  # Initialize it globally

global imgCanvas
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Initial blank state

global cap
cap=None

def gen_frames(callback):
    global target_value
    global imgCanvas
    global cap

    if cap is None:
        return

    detector = htm.HandDetector()

    Previous_Recognized_letter = False

    is_drawing_enabled = True
    prev_index_finger_tip = None

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.8) as hands:
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        prev_index_finger_tip = None
        while True:
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)

            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, False)
                    Fingers = are_all_fingers_up(hand_landmarks)
                    all_fingers_up = Fingers[0]  # True or False
                    Fingers_list = Fingers[1]
                    print(Fingers_list)

                    if all_fingers_up and not Previous_Recognized_letter and np.any(imgCanvas != 0):
                        Previous_Recognized_letter = True
                        callback(imgCanvas, image)

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lmList = [[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in hand_landmarks.landmark]
                    index_finger_tip = lmList[8]

                    fingers = []
                    if lmList and lmList[8][1] < lmList[6][1]:
                        fingers.append(True)
                    else:
                        fingers.append(False)

                    if lmList and lmList[12][1] < lmList[10][1]:
                        fingers.append(True)
                    else:
                        fingers.append(False)

                    is_drawing_enabled = fingers[0] == True and fingers[1] == False

                    if is_drawing_enabled and prev_index_finger_tip is not None:
                        Previous_Recognized_letter = False  # Reset the flag
                        cv2.line(imgCanvas, prev_index_finger_tip, (index_finger_tip[0], index_finger_tip[1]), draw_color, 30)
                        cv2.line(imgCanvas, prev_index_finger_tip, (index_finger_tip[0], index_finger_tip[1]), draw_color, 30)

                    prev_index_finger_tip = (index_finger_tip[0], index_finger_tip[1])

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

            image = cv2.bitwise_and(image, imgInv)
            image = cv2.bitwise_or(image, imgCanvas)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                Previous_Recognized_letter = False
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_drawing(imgCanvas, image):

    if draw_color == (255,255,255):
        print("***************White***************")
        canvas_with_boxes = imgCanvas.copy ()
        # make a copy of the original canvas to apply the bounded boxes on it

        # Convert grayscale to RGB if necessary
        if len (canvas_with_boxes.shape) == 2 or (
                len (canvas_with_boxes.shape) == 3 and canvas_with_boxes.shape [2] == 1):
            canvas_with_boxes = cv2.cvtColor (imgCanvas, cv2.COLOR_GRAY2BGR)
        cv2.imshow ("Image with Rectangles", canvas_with_boxes)
        # Convert canvas to grayscale
        gray = cv2.cvtColor (canvas_with_boxes, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray",gray)
        # Threshold the grayscale image to get a binary image
        _, thresh = cv2.threshold (gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours (thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store all bounding boxes
        bounding_boxes = []

        # Process each contour
        for contour in contours:
            # Find the bounding box enclosing the contour
            x, y, w, h = cv2.boundingRect (contour)

            # Store the bounding box
            bounding_boxes.append ((x, y, x + w, y + h))

            # Draw a rectangle around the contour on the canvas
            cv2.rectangle (canvas_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the bounding box that encloses all smaller rectangles
        x_min = min (box [0] for box in bounding_boxes)
        y_min = min (box [1] for box in bounding_boxes)
        x_max = max (box [2] for box in bounding_boxes)
        y_max = max (box [3] for box in bounding_boxes)

        # Extract the region of interest (ROI) from the canvas using the composite bounding box
        roi = canvas_with_boxes [y_min:y_max, x_min:x_max]

        # Resize the ROI to the expected dimensions
        preprocessed_image = cv2.resize (roi, (32, 32))

        # Display the ROI for verification
        cv2.imshow("ROI", preprocessed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # Normalize the image
        preprocessed_image = preprocessed_image / 255.0

        # Ensure the image has the correct shape (batch_size, height, width, channels)
        preprocessed_image = np.expand_dims (preprocessed_image, axis=0)  # Shape: (1, 32, 32, 3)

        # Predict the class of the drawing along with confidence score
        predicted_class, is_correct = model_handler.predict_with_confidence (preprocessed_image, target_value)
        predicted_class = str (predicted_class)  # convert it to string
        print ('predicted_class: ',predicted_class)

        # Determine the color of the rectangle based on prediction result
        rect_color = (0, 255, 0) if is_correct else (0, 0, 255)

        global isCorrectFlag
        global isNOTCorrectFlag

        isCorrectFlag = is_correct
        isNOTCorrectFlag =not is_correct
        # print('isCorrectFlag set to', isCorrectFlag)

        # Draw a rectangle around the camera feed
        cv2.rectangle (image, (0, 0), (imgCanvas.shape [1], imgCanvas.shape [0]), rect_color, 20)
        # cv2.putText (image, str (predicted_class), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Draw a rectangle around the canvas
        cv2.rectangle (imgCanvas, (0, 0), (image.shape [1], image.shape [0]), rect_color, 30)
        # cv2.putText (imgCanvas, str (predicted_class), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # Wait for a key press to proceed
        cv2.waitKey (500)  # Adjust the delay as needed
        # Show the image with rectangles drawn
        # cv2.imshow("Image with Rectangles", canvas_with_boxes)
    else:
        print("***************Coloured***************")

        canvas_with_boxes = imgCanvas.copy ()

        # Threshold each channel separately to generate binary masks for each color
        _, thresh_r = cv2.threshold (canvas_with_boxes [:, :, 0], 1, 255, cv2.THRESH_BINARY)
        _, thresh_g = cv2.threshold (canvas_with_boxes [:, :, 1], 1, 255, cv2.THRESH_BINARY)
        _, thresh_b = cv2.threshold (canvas_with_boxes [:, :, 2], 1, 255, cv2.THRESH_BINARY)

        # Combine binary masks to create a single mask representing the presence of any color
        combined_mask = cv2.bitwise_or (thresh_r, cv2.bitwise_or (thresh_g, thresh_b))

        # Invert the combined mask to obtain regions where there is no writing
        inverted_mask = cv2.bitwise_not (combined_mask)

        # Fill the regions with white color where there is no writing
        canvas_with_boxes [inverted_mask == 255] = [255, 255, 255]

        # Convert canvas to grayscale with increased threshold value
        gray = cv2.cvtColor (canvas_with_boxes, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold (gray, 200, 255, cv2.THRESH_BINARY)

        # Invert the colors of the gray image
        inverted_gray = cv2.bitwise_not (gray)

        cv2.imshow ("inverted_gray", inverted_gray)
        # Adaptive threshold the inverted gray image to get a binary image
        _, thresh_gray = cv2.threshold (inverted_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours (thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store all bounding boxes
        bounding_boxes = []

        # Process each contour
        for contour in contours:
            # Find the bounding box enclosing the contour
            x, y, w, h = cv2.boundingRect (contour)

            # Store the bounding box
            bounding_boxes.append ((x, y, x + w, y + h))

            # Draw a rectangle around the contour on the canvas
            cv2.rectangle (inverted_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the bounding box that encloses all smaller rectangles
        x_min = min (box [0] for box in bounding_boxes)
        y_min = min (box [1] for box in bounding_boxes)
        x_max = max (box [2] for box in bounding_boxes)
        y_max = max (box [3] for box in bounding_boxes)

        # Extract the region of interest (ROI) from the canvas using the composite bounding box
        roi = inverted_gray [y_min:y_max, x_min:x_max]

        cv2.imshow ("roi", roi)
        # Resize the ROI to the expected dimensions
        preprocessed_image = cv2.resize (roi, (32, 32))

        # Ensure the image has three channels (assuming RGB)
        preprocessed_image = cv2.cvtColor (preprocessed_image, cv2.COLOR_GRAY2RGB)

        cv2.imshow ("preprocessed_image", preprocessed_image)
        # Normalize the image
        preprocessed_image = preprocessed_image / 255.0

        # Ensure the image has the correct shape (batch_size, height, width, channels)
        preprocessed_image = np.expand_dims (preprocessed_image, axis=0)  # Shape: (1, 32, 32, 3)

        # Predict the class of the drawing along with confidence score
        predicted_class, is_correct = model_handler.predict_with_confidence (preprocessed_image, target_value)
        predicted_class = str (predicted_class)  # convert it to string

        # Determine the color of the rectangle based on prediction result
        rect_color = (0, 255, 0) if is_correct else (0, 0, 255)

        isCorrectFlag = is_correct
        isNOTCorrectFlag =not is_correct

        # Draw a rectangle around the camera feed
        cv2.rectangle (image, (0, 0), (imgCanvas.shape [1], imgCanvas.shape [0]), rect_color, 20)
        cv2.putText (image, str (predicted_class), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Draw a rectangle around the canvas
        cv2.rectangle (imgCanvas, (0, 0), (image.shape [1], image.shape [0]), rect_color, 30)
        cv2.putText (imgCanvas, str (predicted_class), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # Show the canvas with rectangles drawn
        cv2.imshow ("Canvas with Rectangles", canvas_with_boxes)

        # Wait for a key press to proceed
        cv2.waitKey (2)  # Adjust the delay as neede


@app.route('/')
def index():
    return render_template('start.html')

@app.route('/numbers_interface')
def index_page3():
    return render_template('numbers_interface.html')

@app.route('/index')
def index_page():
    videos = Video.query.all()  # Get all records
    for video in videos:
        print(f"ID: {video.id}, Name: {video.name}, File Path: {video.file_path}, Description: {video.description}")
    return render_template('index.html',video=videos)  # Pass videos data to template


# Flask route to get a video by ID
@app.route('/video/<int:id>', methods=['GET'])
def get_video_by_id(id):
    video = Video.query.filter_by(id=id).first()  # Get the video with the given ID
    if video:
        return jsonify({
            'name': video.name,
            'file_path': video.file_path,
            'description': video.description
        })
    else:
        return jsonify({'error': 'Video not found'}), 404  # Handle video not found

@app.route('/SelectWritingPreference_Interface')
def index_page2():
    return render_template('SelectWritingPreference_Interface.html')

# @app.route('/clear_video_feed', methods=['POST'])
# def clear_video_feed():
#     global imgCanvas, Previous_Recognized_letter, isCorrectFlag,isNOTCorrectFlag,cap
#     if cap is not None:
#         imgCanvas = np.zeros((720, 1280, 3), np.uint8)
#         Previous_Recognized_letter = False  # Reset the flag
#         isCorrectFlag = False
#         isNOTCorrectFlag = False
#         # print(Previous_Recognized_letter)
#     return jsonify(success=True)

@app.route('/clear_video_feed', methods=['POST'])
def clear_video_feed():
    global imgCanvas, Previous_Recognized_letter, isCorrectFlag, isNOTCorrectFlag, cap

    if cap is not None:
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        Previous_Recognized_letter = False  # Reset the flag
        isCorrectFlag = False
        isNOTCorrectFlag = False
        # print(Previous_Recognized_letter)

    return jsonify(success=True)


@app.route('/change_color', methods=['POST'])
def change_color():
    global draw_color
    color_name = request.json['color']
    colors = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'white': (255, 255, 255),
    }
    draw_color = colors.get(color_name, (255, 255, 255))
    return jsonify(success=True)



@app.route('/set_target_value', methods=['POST'])
def set_target_value():
    global target_value
    data = request.get_json()  # Make sure you're calling get_json()
    print('Data received:', data)  # Check the data received from the POST request
    target_value = data.get('target_value')
    print('target_value set to:', target_value)  # Check the value after setting
    return jsonify(success=True)



@app.route('/check_correct')
def check_correct():
    global isCorrectFlag, isNOTCorrectFlag
    # print('is_correct set to', isCorrectFlag)
    return jsonify(is_correct=bool(isCorrectFlag), is_not_correct=bool(isNOTCorrectFlag))

@app.route('/open_camera', methods=['POST'])
def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    return jsonify({'message': 'Camera access granted by user'})


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(process_drawing), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)



