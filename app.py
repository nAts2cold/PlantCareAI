import os
import secrets
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = secrets.token_hex(16)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'images'), exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global model variable
model = None

# 38 Classes based on the generic dataset structure
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def load_model():
    global model
    model_path = 'final_plant_disease_classifier.keras'
    if not os.path.exists(model_path):
        model_path = 'final_plant_disease_classifier.keras' # Fallback to the other model if final doesn't exist
    
    if not os.path.exists(model_path):
        print(f"Warning: Neither model could be found at {os.getcwd()}")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e:
        print(f"Error loading model: {e}")

def predict_image(filepath):
    # Load and preprocess image (Assuming MobileNetV2 224x224 input)
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class_idx])
    
    label = CLASS_NAMES[predicted_class_idx]
    
    # Clean up the label for better display
    label = label.replace("___", " - ").replace("_", " ")

    return {"class": label, "confidence": confidence}

@app.route('/')
def upload():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected.')
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            if model is None:
                raise Exception("Model is not loaded. Please restart the server or verify model files.")
                
            prediction = predict_image(filepath)
            
            # Save image to static folder for display
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], 'images', static_filename)
            Image.open(filepath).save(static_path)
            
            # Store in session
            session['prediction'] = prediction['class']
            session['confidence'] = f"{prediction['confidence'] * 100:.2f}%"
            # Using forward slashes for URLs, assuming static folder route
            session['image_url'] = url_for('static', filename=f'images/{static_filename}')
            
            os.remove(filepath) # Clean up temp file
            
            # Redirect to result
            return redirect(url_for('result'))
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('index.html', error=f"An error occurred: {str(e)}")

@app.route('/result')
def result():
    prediction = session.get('prediction')
    confidence = session.get('confidence')
    image_url = session.get('image_url')
    
    if not prediction:
        return redirect(url_for('upload'))
        
    return render_template('result.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)