import os
import pandas as pd
import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import cv2
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tf.keras.callbacks import LearningRateScheduler


# Функция для загрузки и предобработки изображений
def load_and_preprocess_image(file_path):
    # Загрузка изображения
    image = cv2.imread(file_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {file_path}")
        return None
    # Изменение размера изображения до 50x50 пикселей
    # image = cv2.resize(image, (50, 50))
    # Нормализация значений пикселей
    image = image / 255.0
    return image


# Загрузка данных из CSV-файла
print('Загрузка данных из CSV-файла')
data = pd.read_csv('FULL FULL FULL.csv')

# Разделение данных на признаки (x) и метки (y)
print('Разделение данных на признаки (x) и метки (y)')
x = data['filename']  # Признаки (названия файлов)
y = data['label']  # Метки классов

# Разделение данных на тренировочный и тестовый наборы
print('Разделение данных на тренировочный и тестовый наборы')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Путь к папке с изображениями
images_directory = "..."
#
# # Подготовка массивов изображений для обучения и тестирования
# print('Подготовка массивов изображений для обучения и тестирования')
# X_train_images = np.array(
#     [load_and_preprocess_image(os.path.join(images_directory, file_path)) for file_path in x_train])
# X_test_images = np.array([load_and_preprocess_image(os.path.join(images_directory, file_path)) for file_path in x_test])
#
# # Сохранение массивов в файлы
# print('Сохранение массивов X в файлы')
# np.save('X_train_images.npy', X_train_images)
# np.save('X_test_images.npy', X_test_images)

# Создание словаря для сопоставления строковых меток классов с числовыми идентификаторами
unique_classes = ['A', 'B', 'C', 'CH', 'D', 'E', 'EE', 'F', 'G', 'I', 'K', 'L', 'M', 'MZ', 'N', 'O', 'P', 'R', 'SH',
                  'T', 'TS', 'U', 'V', 'X', 'Y', 'YA', 'YY', 'Z', 'ZH']
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
#
# # # Преобразование меток в числовой формат
# y_train_cat = np.array([class_to_index[label] for label in y_train])
# y_test_cat = np.array([class_to_index[label] for label in y_test])
#
# # Преобразование меток в формат one-hot
# print('Преобразование меток в формат one-hot')
# y_train_cat = to_categorical(y_train_cat, num_classes=29)
# y_test_cat = to_categorical(y_test_cat, num_classes=29)
#
# # Сохранение массивов в файлы
# print('Сохранение массивов Y в файлы')
# np.save('y_train_cat.npy', y_train_cat)
# np.save('y_test_cat.npy', y_test_cat)
# #
# Загрузка массивов из файлов
X_train_images = np.load('X_train_images.npy')
X_test_images = np.load('X_test_images.npy')
y_train_cat = np.load('y_train_cat.npy')
y_test_cat = np.load('y_test_cat.npy')

# Создание модели
print('Создание модели')

# 531
model = Sequential()
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(50, 50, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(29, activation='softmax'))


# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Обучение модели
history = model.fit(X_train_images, y_train_cat, epochs=12, batch_size=32, validation_data=(X_test_images, y_test_cat))

# Сохранение модели
model.save("93k_colored_with_bg_v7.keras")

number = "new3"
#
# График валидационной точности
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')  # Название оси X
plt.ylabel('Accuracy')  # Название оси Y
plt.title('Точность на тестовом наборе')  # Заголовок графика
plt.grid(True)  # Включаем сетку
plt.legend()  # Добавляем легенду
plt.savefig("validation_accuracy " + number + ".jpg")  # Сохранение графика как JPG
plt.show()
#
# График тренировочной точности
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Точность на тренировочном наборе')
plt.grid(True)
plt.legend()
plt.savefig("training_accuracy " + number + ".jpg")
plt.show()
#
# График потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Значения функции потерь на тренировочном наборе')
plt.grid(True)
plt.legend()
plt.savefig("training_loss " + number + ".jpg")
plt.show()

# График потерь
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Значения функции потерь на тестовом наборе')
plt.grid(True)
plt.legend()
plt.savefig("validation_loss " + number + ".jpg")
plt.show()

#
# График сравнения валидационной и тренировочной точности
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Точность на тестовом и тренировочном наборах')
plt.grid(True)
plt.legend()
plt.savefig("validation_vs_training_accuracy " + number + ".jpg")
plt.show()
#
# График потерь
# plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Значения функции потерь на тестовом и тренировочном наборах')
plt.grid(True)
plt.legend()
plt.savefig("validation_vs_training_loss " + number + ".jpg")
plt.show()
print("Loss: ", history.history['loss'], "Val loss: ", history.history['val_loss'])
# #
# # График точности
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Train vs Validation Accuracy')
# plt.grid(True)
# plt.legend()
# plt.show()
# print("Accuracy: ", history.history['accuracy'], "Val acc: ", history.history['val_accuracy'])
