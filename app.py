from flask import Flask, render_template, request, redirect
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Flask uygulamasını başlat
app = Flask(__name__)

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('C:/Users/yefe/Desktop/Proje/models/best_cnn_model.keras')

# Ana sayfa route
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    try:
        # Dosyayı BytesIO'ya çevir
        img = Image.open(BytesIO(file.read()))  # PIL Image
        img = img.resize((128, 128))  # Görseli modelin istediği boyutta yeniden boyutlandır
        img_array = img_to_array(img) / 255.0  # Normalize et
        img_array = np.expand_dims(img_array, axis=0)  # Modelin beklediği şekle sok
        
        # Model ile tahmin yap
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Sınıf isimleri
        class_names = ['Cataract', 'Diabetic Retinopathy',  'Glaucoma', 'Normal']

        
        # Tahmin sonucu metni
        prediction_text = f"The model predicts: {class_names[predicted_class[0]]}"
        
        # Sonuçları index.html'e gönder
        return render_template('index.html', prediction_text=prediction_text)
    
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
