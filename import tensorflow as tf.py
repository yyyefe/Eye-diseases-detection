import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Verilerin Hazırlanması
data_dir = "C:/Users/yefe/Downloads/archive (1)/dataset"  # Veri seti klasörünün yolu
img_height, img_width = 128, 128  # Görselleri 128x128 boyutuna yeniden boyutlandırma

# Görselleri ve etiketleri yüklemek için listeler oluştur
images = []
labels = []

# Her sınıf için görselleri yükle
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):  # Eğer bir klasörse
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Görseli yükle ve boyutlandır
                img = load_img(img_path, target_size=(img_height, img_width))
                # Görseli numpy array'e dönüştür ve normalize et (0-1 aralığına)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(class_name)  # Sınıf ismini etiket olarak al
            except Exception as e:
                print(f"Görsel yükleme hatası: {img_path}, Hata: {e}")

# Görselleri ve etiketleri numpy array'e dönüştür
images = np.array(images)
labels = np.array(labels)

# Etiketleri sayısal değerlere dönüştürmek için LabelEncoder kullan
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Etiketleri one-hot encode formatına çevir (4 sınıf için)
labels = to_categorical(labels, num_classes=4)

# Eğitim ve doğrulama setlerini ayır (verilerin %20'si doğrulama için)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 2. Veri Artırma
# Eğitim seti için veri artırma (daha çeşitli veri üretmek için yatay çevirme ve döndürme)
train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10)
# Doğrulama seti için veri artırma yok
val_datagen = ImageDataGenerator()

# Veri artırmayı uygulanmış veri akışları oluştur
train_data = train_datagen.flow(X_train, y_train, batch_size=32)
val_data = val_datagen.flow(X_val, y_val, batch_size=32)

# 3. CNN Modeli Oluşturma
model = Sequential([
    # İlk evrişim katmanı ve maksimum havuzlama
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Aşırı öğrenmeyi azaltmak için dropout

    # İkinci evrişim katmanı ve maksimum havuzlama
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Üçüncü evrişim katmanı ve maksimum havuzlama
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Verileri düzleştir ve tam bağlı katmanları ekle
    Flatten(),
    Dense(128, activation='relu'),  # 128 nöronlu tam bağlı katman
    Dropout(0.5),  # Daha fazla dropout
    Dense(4, activation='softmax')  # 4 sınıf için softmax aktivasyonu
])

# 4. Modeli Derleme
# Adam optimizasyon algoritması, kategorik çapraz entropi kaybı ve doğruluk metriği ile model derlenir
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. ModelCheckpoint ile en iyi modeli kaydetme
# 'models' klasörünü oluştur (yoksa)
os.makedirs("models", exist_ok=True)
# En iyi modeli val_loss değerine göre kaydet
model_checkpoint = ModelCheckpoint("models/best_cnn_model.keras", save_best_only=True, monitor='val_loss')

# 6. Modeli Eğitme
history = model.fit(
    train_data,  # Eğitim verileri
    epochs=20,  # 20 epoch boyunca eğit
    validation_data=val_data,  # Doğrulama verileri
    callbacks=[model_checkpoint]  # ModelCheckpoint callback'i
)

# 7. Modeli Değerlendirme
# Doğrulama verileri üzerinde modelin performansını değerlendir
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Doğrulama kaybı: {val_loss:.4f}, Doğrulama doğruluğu: {val_accuracy:.4f}")

# 8. Karışıklık Matrisi ve F1 Skoru
# Doğrulama seti üzerindeki tahminler
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)  # Tahmin edilen sınıflar
y_val_true_classes = np.argmax(y_val, axis=1)  # Gerçek sınıflar

# Karışıklık Matrisi
cm = confusion_matrix(y_val_true_classes, y_val_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Karışıklık Matrisi")
plt.show()

# F1 Skoru
f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')
print(f"Makro F1 Skoru: {f1:.4f}")

# Sınıflandırma Raporu
report = classification_report(y_val_true_classes, y_val_pred_classes, target_names=label_encoder.classes_)
print("\nSınıflandırma Raporu:\n", report)

# 9. Kaydedilen Modeli Yükleme
# En iyi modeli yeniden yükle
best_model = tf.keras.models.load_model("models/best_cnn_model.keras")
print("En iyi model başarıyla yüklendi ve kullanılabilir.")
