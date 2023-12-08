# import cv2
# import numpy as np
# from keras.models import model_from_json


# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # load json and create model
# json_file = open('D:\Python Program\Design Project\CCA Project\Emotion_detection_with_CNN-main\model\emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # load weights into new model
# emotion_model.load_weights("D:\Python Program\Design Project\CCA Project\Emotion_detection_with_CNN-main\model\emotion_model.h5")
# print("Loaded model from disk")
# cap = cv2.VideoCapture(0)

# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     if not ret:
#         break
#     face_detector = cv2.CascadeClassifier('D:\Python Program\Design Project\CCA Project\Emotion_detection_with_CNN-main\haarcascades\haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import model_from_json

# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# # Load json and create model
# json_file = open('D:/Python Program/Design Project/CCA Project/Emotion_detection_with_CNN-main/model/New folder/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# # Load weights into the new model
# emotion_model.load_weights('D:/Python Program/Design Project/CCA Project/Emotion_detection_with_CNN-main/model/New folder/emotion_model.h5')
# print("Loaded model from disk")

# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
#     while True:
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (1280, 720))
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(frame_rgb)

#         if results.detections:
#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, _ = frame.shape
#                 x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
#                 roi_gray_frame = frame[y:y + h, x:x + w]
#                 if not roi_gray_frame.size:
#                     continue  # Skip this frame if no face is detected
#                 # Convert the image to grayscale
#                 roi_gray_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_BGR2GRAY)
#                 cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#                 # Predict the emotions
#                 emotion_prediction = emotion_model.predict(cropped_img)
#                 maxindex = int(np.argmax(emotion_prediction))
#                 cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         cv2.imshow('Emotion Detection', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

#Code for CCA music recommendation
import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import os
import pygame
import random

# Define the path to the emotion model files
emotion_model_path = 'C:/Users/hp/Desktop/PROJECTS23/Projects2k23-master/backend/venv/EmotionDetector/model/'
json_file_path = os.path.join(emotion_model_path, 'emotion_model.json')
weights_file_path = os.path.join(emotion_model_path, 'emotion_model.h5')

# Load json and create model
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the new model
emotion_model.load_weights(weights_file_path)
print("Loaded model from disk")

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Sad", 4: "Surprised"}

# Initialize Pygame for audio playback
pygame.mixer.init()

# Define the path to the folder containing music for each emotion
music_folder = 'C:/Users/hp/Desktop/PROJECTS23/Projects2k23-master/backend/venv/EmotionDetector/Music/'  # Update this path to your music folder

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Variables for capturing image once
capture_image = False
image_captured = False

# Face detection context
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                roi_gray_frame = frame[y:y + h, x:x + w]

                if not roi_gray_frame.size:
                    continue  # Skip this frame if no face is detected

                # Convert the image to grayscale
                roi_gray_frame = cv2.cvtColor(roi_gray_frame, cv2.COLOR_BGR2GRAY)
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # Predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                detected_emotion = emotion_dict[maxindex]

                # Play a random song from the corresponding emotion folder
                emotion_music_folder = os.path.join(music_folder, detected_emotion)

                if not pygame.mixer.music.get_busy():
                    songs = os.listdir(emotion_music_folder)
                    if songs:
                        song_to_play = os.path.join(emotion_music_folder, random.choice(songs))
                        pygame.mixer.music.load(song_to_play)
                        pygame.mixer.music.play()

                        # Capture image once when a face is detected and song starts
                        capture_image = True

                cv2.putText(frame, detected_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Capture image only once when capture_image is True
        if capture_image and not image_captured:
            cv2.imwrite('captured_image.jpg', frame)
            print("Image Captured!")
            image_captured = True

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
