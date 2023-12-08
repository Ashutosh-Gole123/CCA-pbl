from flask import Flask, request, jsonify
import os
import base64
import uuid
import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json
import pygame
import random

app = Flask(__name__)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        uploaded_file = request.files['photo_data']
        print('Received data:', upload_file)

        if uploaded_file:
            # Assuming you want to save the file in a 'uploads' folder
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # You can do further processing here if needed

            return jsonify({'message': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# @app.route('/api/uploadImage', methods=['POST'])
# def upload_image():
#     try:
#         # Get the image data from the request form parameter
#         image_data = request.form.get('image_data')
#         print('Received data:', image_data)

#         if image_data:
#             # Decode the base64 image data
#             image_binary = base64.b64decode(image_data)
#             print('Length of decoded image binary:', len(image_binary))

#             # Save the image to the uploads folder with a unique filename
#             upload_folder = 'uploads'
#             if not os.path.exists(upload_folder):
#                 os.makedirs(upload_folder)

#             # Generate a unique filename using uuid
#             image_filename = str(uuid.uuid4()) + '.png'
#             image_path = os.path.join(upload_folder, image_filename)

#             with open(image_path, 'wb') as image_file:
#                 image_file.write(image_binary)

#             return jsonify({'message': 'Image uploaded successfully', 'filename': image_filename})

#     except Exception as e:
#         print('Error:', e)
#         return jsonify({'error': str(e)}), 500


@app.route('/api/uploadImage', methods=['POST'])
def upload_image():
    try:
        # Get the image data from the request form parameter
        image_data = request.form.get('image_data')
        print('Received data',image_data)

        if image_data:
            # Decode the base64 image data
            image_binary = base64.b64decode(image_data)
            print("Image data: ", image_binary )
            print('Length of decoded image binary:', len(image_binary))

            # Save the image to the uploads folder with a unique filename
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Generate a unique filename using uuid
            image_filename = str(uuid.uuid4()) + '.png'
            image_path = os.path.join(upload_folder, image_filename)

            with open(image_path, 'wb') as image_file:
                image_file.write(image_binary)

            # Call the function to perform emotion detection on the uploaded image
            emotion_detection_on_image(image_path)

            return jsonify({'message': 'Image uploaded successfully', 'filename': image_filename})

    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 500

def emotion_detection_on_image(image_path):
    # ... (your existing code for model loading, Pygame initialization, etc.)
     # Define the path to the emotion model files
    emotion_model_path = './EmotionDetector/model'
    #emotion_model_path = './EmotionDetector/model'
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

    #    Initialize Pygame for audio playback
    pygame.mixer.init()

    # Define the path to the folder containing music for each emotion
    music_folder = './EmotionDetector/Music/'  # Update this path to your music folder
  
   # music_folder = './EmotionDetector/Music/'  # Update this path to your music folder

    # Initialize MediaPipe for face detection
    mp_face_detection = mp.solutions.face_detection
  

    # Read the uploaded image
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (1280, 720))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection context
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
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
                print("Emotion detected: ",detected_emotion)
                # Play a random song from the corresponding emotion folder
                emotion_music_folder = os.path.join(music_folder, detected_emotion)
                print("Emotion music folder: ",emotion_music_folder)
                if not pygame.mixer.music.get_busy():
                    songs = os.listdir(emotion_music_folder)
                    if songs:
                        song_to_play = os.path.join(emotion_music_folder, random.choice(songs))
                        # pygame.mixer.music.load(song_to_play)
                        # pygame.mixer.music.play()
                        # Check if the file exists before attempting to load and play it
                        if os.path.exists(song_to_play):
                            pygame.mixer.music.load(song_to_play)
                            pygame.mixer.music.play()
                        else:
                            print("File does not exist: ", song_to_play)

                        # Capture image once when a face is detected and song starts
                        cv2.putText(frame, detected_emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imwrite('captured_image.jpg', frame)
                        print("Image Captured!")

    # Display the image with emotion text
    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
