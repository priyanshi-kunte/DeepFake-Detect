import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model once at startup
print("Loading model...")
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import layers
import tensorflow as tf

class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape if shape != -1 else tf.shape(inputs)[i]
                     for i, shape in enumerate(self.noise_shape)])

model = load_model('./tmp_checkpoint/best_model.h5', custom_objects={
    'FixedDropout': FixedDropout
})
detector = MTCNN()
print("Model loaded!")

INPUT_SIZE = 128

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def predict_image(img_array):
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    
    results = []
    
    if len(faces) == 0:
        return [{"error": "No face detected in image"}]
    
    for i, face in enumerate(faces):
        confidence = face['confidence']
        if confidence < 0.95:
            continue
        x, y, w, h = face['box']
        margin = int(0.3 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_array.shape[1], x + w + margin)
        y2 = min(img_array.shape[0], y + h + margin)
        
        face_crop = img_array[y1:y2, x1:x2]
        processed = preprocess_face(face_crop)
        prediction = model.predict(processed)[0][0]
        
        label = "REAL" if prediction >= 0.5 else "FAKE"
        prob = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        
        results.append({
            "face": i + 1,
            "label": label,
            "confidence": round(prob * 100, 2)
        })
    
    if len(results) == 0:
        return [{"error": "No face detected with high enough confidence"}]
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    filename = file.filename.lower()
    
    # Handle image
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = predict_image(img)
        return jsonify({"type": "image", "results": results})
    
    # Handle video
    elif filename.endswith(('.mp4', '.avi', '.mov')):
        video_path = './tmp_upload.mp4'
        file.save(video_path)
        
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(5)
        all_predictions = []
        
        while cap.isOpened():
            frame_id = cap.get(1)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % max(1, int(frame_rate)) == 0:
                results = predict_image(frame)
                for r in results:
                    if 'label' in r:
                        all_predictions.append(r['label'])
        
        cap.release()
        os.remove(video_path)
        
        if len(all_predictions) == 0:
            return jsonify({"type": "video", "results": [{"error": "No faces detected in video"}]})
        
        fake_count = all_predictions.count('FAKE')
        real_count = all_predictions.count('REAL')
        total = len(all_predictions)
        final_label = "FAKE" if fake_count > real_count else "REAL"
        
        return jsonify({
            "type": "video",
            "results": [{
                "label": final_label,
                "confidence": round((max(fake_count, real_count) / total) * 100, 2),
                "frames_analyzed": total,
                "fake_frames": fake_count,
                "real_frames": real_count
            }]
        })
    
    else:
        return jsonify({"error": "Unsupported file type. Use jpg, png, mp4"})

if __name__ == '__main__':
    app.run(debug=True)