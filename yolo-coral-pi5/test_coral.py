import cv2
import time
import threading
from ultralytics import YOLO

class VideoStream:
    """Класс для многопоточного чтения кадров с веб-камеры"""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def main():
    #MODEL_PATH = "models/best_full_integer_quant_edgetpu.tflite"
    MODEL_PATH = "models/yolov8n_full_integer_quant_edgetpu.tflite"
    print(f"[INFO] Загрузка модели {MODEL_PATH}...")
    
    # Загружаем модель. Ultralytics сама найдет Edge TPU
    model = YOLO(MODEL_PATH, task='detect')

    print("[INFO] Запуск веб-камеры...")
    # Инициализируем многопоточный поток камеры
    vs = VideoStream(src=0, width=640, height=480).start()
    time.sleep(2.0) # Даем камере прогреться

    prev_time = time.time()
    fps_avg = 0

    print("[INFO] Готово. Нажмите 'q' для выхода.")
    
    while True:
        frame = vs.read()
        if frame is None:
            continue

        # Замеряем только время работы нейросети (Coral)
        start_inf = time.time()
        #results = model.predict(source=frame, conf=0.4, verbose=False)
        results = model.predict(source=frame, conf=0.4, classes=[0], verbose=False)
        inf_time = time.time() - start_inf
        
        coral_fps = 1 / inf_time if inf_time > 0 else 0

        # Отрисовка рамок (работает на CPU Raspberry)
        annotated_frame = results[0].plot()

        # Общий FPS системы (с учетом отрисовки)
        curr_time = time.time()
        sys_fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        fps_avg = 0.9 * fps_avg + 0.1 * sys_fps

        # Выводим обе цифры
        cv2.putText(annotated_frame, f"System FPS: {fps_avg:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Coral FPS: {coral_fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Raspberry Pi 5 + Google Coral", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Завершение работы...")
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()