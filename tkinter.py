from PIL import Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
import time
import mediapipe as mp
from keras.models import load_model

# Создание основного окна
root = tk.Tk()
root.title("Gesture")
root.geometry("800x600")  # Установите ширину 800 пикселей и высоту 600 пикселей
# Инициализация захвата с камеры
cap = cv2.VideoCapture(0)

# Словарь для сопоставления числовых индексов классов с соответствующими буквенными метками
class_to_label = {0: 'А', 1: 'Б', 2: 'С', 3: 'Ч', 4: 'Д', 5: 'Е', 6: 'Э', 7: 'Ф', 8: 'Г', 9: 'И', 10: 'К',
                  11: 'Л', 12: 'М', 13: 'Ь', 14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'Ш', 19: 'Т', 20: 'Ц',
                  21: 'Ю', 22: 'В', 23: 'Х', 24: 'У', 25: 'Я', 26: 'Ы', 27: 'З', 28: 'Ж'}


model = load_model("93k_colored_with_bg_v7.keras")

previos = "-"
prevprev = "-"
message = tk.Text(root, width=35, height=1, bd=5, relief="solid")  # Ширина - 50 символов, высота - 10 строк
message.place(x=10, y=500)
message.configure(font=("Consolas", 30))
# Создание виджета Label для отображения изображения
img = tk.Label(root, bd=5, relief="solid")
img.place(x=10, y=10)
gest = tk.Text(root, width=1, height=1, bd=5, relief="solid")
gest.place(x=700, y=10)
gest.configure(font=("Consolas", 30))
previos_time = int(time.time() * 1000)
current_time = int(time.time() * 1000)
last_hand_time = int(time.time() * 1000)

started = False
paused = False
last_gest = "0"
hand_flag = False
dot = False
space = False
time_main = 1000
time_second = 2000
flipped = False


def detect_hand(image):
    # Инициализация объекта для обнаружения рук
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Обнаружение рук на изображении
    results = hands.process(image)

    # Проверка, есть ли обнаруженные руки
    if results.multi_hand_landmarks:
        # Если обнаружены руки, возвращаем 1
        hands.close()
        return 1
    else:
        # Если руки не обнаружены, возвращаем 0
        hands.close()
        return 0


# Функция для чтения кадра с камеры и отображения его в Tkinter
def update_frame():
    global previos, prevprev, started, previos_time, current_time, last_gest, hand_flag, last_hand_time, dot, space, time_main, time_second

    if not started:
        message.replace("1.0", "1.35", "                                   ")
        started = True

    ret, frame = cap.read()  # Чтение кадра с камеры
    if flipped:
        frame = cv2.flip(frame, 1)
    buf = frame
    frame_rgb = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)  # Преобразование цветовой модели OpenCV BGR в RGB
    hand = detect_hand(frame_rgb)

    # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    hand_flag = hand
    frame = cv2.resize(frame_rgb, (70, 50))
    frame = frame[:, 10:-10]
    img_array = frame / 255.0
    # Расширение размерности изображения (добавление размерности батча)
    current_hand_time = int(time.time() * 1000)

    if hand:
        last_hand_time = int(time.time() * 1000)
        dot = False
        space = False
        image = np.expand_dims(img_array, axis=0)
        # Прогноз с использованием модели
        prediction = model.predict(image, verbose=0)
        # Преобразование предсказания в человекочитаемый формат
        predicted_class_index = np.argmax(prediction)
        predicted_label = class_to_label[predicted_class_index]
        gest.replace("1.0", "1.1", predicted_label)

        if not paused:
            if previos != predicted_label:
                # last_gest = previos
                previos = predicted_label
                previos_time = int(time.time() * 1000)
            current_time = int(time.time() * 1000)

            delta_time = current_time - previos_time

            if delta_time >= time_main:
                if previos not in ["Е", "И", "Ш", "Ь"]:
                    message.delete("1.0", "1.1")
                    message.insert(tk.END, previos)
                    last_gest = previos
                    previos_time = current_time
                if previos in ["Е", "И", "Ш", "Ь"] and delta_time >= time_second:
                    buf_let = ''
                    if previos == 'Ш':
                        buf_let = 'Щ'
                    elif previos == 'Е':
                        buf_let = 'Ё'
                    elif previos == 'И':
                        buf_let = 'Й'
                    elif previos == 'Ь':
                        buf_let = 'Ъ'
                    previos = buf_let
                    previos_time = current_time
                    message.delete("1.34", "1.35")
                    message.insert(tk.END, buf_let)
                    last_gest = previos
                elif previos in ["Е", "И", "Ш", "Ь"]:
                    # if message.get("end-2c") not in ["Е", "И", "Ш", "Ь"]:
                    if last_gest not in ["Е", "И", "Ш", "Ь"]:
                        # previos_time = current_time
                        message.delete("1.0", "1.1")
                        message.insert(tk.END, previos)
                        last_gest = previos
    else:
        last_gest = "1"
        gest.replace("1.0", "1.1", " ")
        previos_time = int(time.time() * 1000)

    if not paused:
        if hand_flag is True:
            last_hand_time = int(time.time() * 1000)

        if space is not True:
            if current_hand_time - last_hand_time > 2000:
                if message.get("end-2c") != " " and message.get("end-2c") != "." and len(
                        message.get("1.0", "1.35")) != "":
                    message.delete("1.0", "1.1")
                    message.insert(tk.END, " ")
                    space = True

        if dot is not True:
            if current_hand_time - last_hand_time > 5000:
                if message.get("end-2c") != "." and message.get("end-2c") != "" and message.get("1.0",
                                                                                                "1.35") != "                                   ":
                    message.delete("1.34", "1.35")
                    # message.delete("1.0", "1.1")
                    message.insert(tk.END, ".")
                    dot = True
    image = Image.fromarray(frame_rgb)  # Создание объекта изображения из массива NumPy
    photo = ImageTk.PhotoImage(image)  # Создание объекта PhotoImage для отображения в Tkinter
    img.config(image=photo)  # Обновление изображения на виджете Label
    img.image = photo  # Сохранение ссылки на объект PhotoImage, чтобы он не был уничтожен сборщиком мусора
    root.after(1, lambda: update_frame())


def clear():
    global message
    message.replace("1.0", "1.35", "                                   ")


def delete_last():
    global message
    message.delete("1.34", "1.35")
    message.insert("1.1", " ")


def toggle_paused():
    global paused
    paused = not paused


def flip():
    global flipped
    flipped = not flipped


btn_paused = tk.Button(text="Pause", command=toggle_paused)
btn_paused.place(x=700, y=120)
btn_deleted = tk.Button(text='Delete', command=delete_last)
btn_deleted.place(x=700, y=170)
btn_clear = tk.Button(text='Clear', command=clear)
btn_clear.place(x=700, y=270)
btn_flip = tk.Button(text='Flip', command=flip)
btn_flip.place(x=700, y=80)

# Вызов функции update_frame для первоначального отображения изображения
update_frame()
# Запуск основного цикла обработки событий
root.mainloop()

cap.release()
