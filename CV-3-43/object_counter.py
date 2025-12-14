import cv2
import numpy as np
import argparse
import time
from collections import defaultdict
import os
import sys

from CV_2_04 import ColorMask, create_red_mask, create_blue_mask, apply_mask
from CV_2_43 import find_and_draw_contours

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class ObjectCounter:
    """Класс для подсчёта объектов по цвету"""
    
    def __init__(self, min_area=1000, max_area=20000):
        self.min_area = min_area
        self.max_area = max_area
        
        self.color_masks = {
            'red': create_red_mask(),
            'blue': create_blue_mask(),
            'green': self._create_green_mask(),
            'yellow': self._create_yellow_mask(),
            'orange': self._create_orange_mask()
        }
        
        self.display_colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255)
        }
        
        self.frame_count = 0
        self.count_history = defaultdict(list)
        self.current_counts = defaultdict(int)
        
        print(f"Инициализирован счетчик объектов:")
        print(f"  Минимальная площадь: {min_area} пикселей")
        print(f"  Максимальная площадь: {max_area} пикселей")
        print(f"  Отслеживаемые цвета: {', '.join(self.color_masks.keys())}")
    
    def _create_green_mask(self):
        """Создание маски для зеленого цвета через ColorMask"""
        mask = ColorMask("green")
        mask.add_hsv_range((np.array([40, 50, 50]), np.array([80, 255, 255])))
        return mask
    
    def _create_yellow_mask(self):
        """Создание маски для желтого цвета через ColorMask"""
        mask = ColorMask("yellow")
        mask.add_hsv_range((np.array([20, 100, 100]), np.array([30, 255, 255])))
        return mask
    
    def _create_orange_mask(self):
        """Создание маски для оранжевого цвета через ColorMask"""
        mask = ColorMask("orange")
        mask.add_hsv_range((np.array([10, 100, 100]), np.array([20, 255, 255])))
        return mask
    
    def process_frame(self, frame):
        """Обработка одного кадра видео"""
        self.frame_count += 1
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        contours_by_color = defaultdict(list)
        
        for color_name, color_mask in self.color_masks.items():
            mask = color_mask.create_mask(hsv_frame)
            
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            masked_frame = apply_mask(frame, mask)
            
            try:
                processed_image, _ = find_and_draw_contours(
                    masked_frame, 
                    self.min_area, 
                    self.max_area
                )
                
                if processed_image is not None:
                    processed_hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
                    lower_green = np.array([40, 40, 40])
                    upper_green = np.array([80, 255, 255])
                    contour_mask = cv2.inRange(processed_hsv, lower_green, upper_green)
                    
                    color_contours, _ = cv2.findContours(
                        contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    filtered_contours = []
                    for contour in color_contours:
                        area = cv2.contourArea(contour)
                        if self.min_area <= area <= self.max_area:
                            filtered_contours.append(contour)
                    
                    color_contours = filtered_contours
                else:
                    color_contours = []
                    
            except Exception as e:
                print(f"Ошибка при использовании обработки контуров: {e}")
            
            if color_contours:
                contours_by_color[color_name] = color_contours
        
        for color_name, contours in contours_by_color.items():
            self.current_counts[color_name] = len(contours)
            self.count_history[color_name].append(len(contours))
        
        result_frame = self._draw_results(frame, contours_by_color)
        
        return result_frame, dict(self.current_counts)
    
    def _draw_results(self, frame, contours_by_color):
        """Рисует обнаруженные объекты и счетчики на кадре"""
        result_frame = frame.copy()
        
        for color_name, contours in contours_by_color.items():
            display_color = self.display_colors.get(color_name, (255, 255, 255))
            
            if len(contours) > 0:
                cv2.drawContours(result_frame, contours, -1, display_color, 2)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), display_color, 2)
                
                label = f"{color_name[0].upper()}{i+1}"
                cv2.putText(result_frame, label, (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        
        result_frame = self._draw_counters(result_frame)
        
        return result_frame
    
    def _draw_counters(self, frame):
        """Отображает счетчики объектов на кадре"""
        _, width = frame.shape[:2]
        
        panel_height = 180
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        title = "OBJECT COUNTER"
        cv2.putText(panel, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frame_info = f"Frame: {self.frame_count}"
        cv2.putText(panel, frame_info, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset = 70
        total_count = 0
        
        for _, (color_name, count) in enumerate(sorted(self.current_counts.items())):
            total_count += count
            
            display_color = self.display_colors.get(color_name, (255, 255, 255))
            
            cv2.circle(panel, (20, y_offset - 5), 8, display_color, -1)
            cv2.circle(panel, (20, y_offset - 5), 8, (255, 255, 255), 1)
            
            text = f"{color_name.capitalize()}: {count} object(s)"
            cv2.putText(panel, text, (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.count_history[color_name] and len(self.count_history[color_name]) > 0:
                recent_counts = self.count_history[color_name][-10:]
                avg_count = np.mean(recent_counts) if recent_counts else 0
                bar_width = int(avg_count * 5)
                bar_width = min(bar_width, 200)
                cv2.rectangle(panel, (250, y_offset - 15), 
                            (250 + bar_width, y_offset + 5),
                            display_color, -1)
            
            y_offset += 25
        
        result = np.vstack([frame, panel])
        return result
    
    def save_statistics(self, filename="object_statistics.txt"):
        """Сохраняет статистику в текстовый файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("СТАТИСТИКА ПОДСЧЕТА ОБЪЕКТОВ ПО ЦВЕТУ\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Всего обработано кадров: {self.frame_count}\n")
            f.write("РЕЗУЛЬТАТЫ ПО ЦВЕТАМ:\n")
            f.write("-" * 60 + "\n")
            
            for color_name in self.color_masks.keys():
                if color_name in self.count_history:
                    counts = self.count_history[color_name]
                    if counts:
                        avg_count = np.mean(counts) if counts else 0
                        max_count = np.max(counts) if counts else 0
                        min_count = np.min(counts) if counts else 0
                        
                        f.write(f"\n{color_name.upper()}:\n")
                        f.write(f"  Среднее количество: {avg_count:.2f}\n")
                        f.write(f"  Максимальное количество: {max_count}\n")
                        f.write(f"  Минимальное количество: {min_count}\n")
                        f.write(f"  Всего обнаружено: {sum(counts)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("ОЦЕНКА ТОЧНОСТИ:\n")
            f.write("-" * 60 + "\n")

            for color_name in self.color_masks.keys():
                if color_name in self.count_history and len(self.count_history[color_name]) > 10:
                    recent_counts = self.count_history[color_name][-10:]
                    std_dev = np.std(recent_counts) if len(recent_counts) > 1 else 0
                    stability = max(0, 100 - (std_dev * 10)) if std_dev > 0 else 100
                    
                    f.write(f"\n{color_name.upper()}:\n")
                    f.write(f"  Стабильность подсчета: {stability:.1f}%\n")
                    f.write(f"  Стандартное отклонение (последние 10 кадров): {std_dev:.2f}\n")
        
        print(f"\nСтатистика сохранена в файл: {filename}")
        print(f"Всего обработано кадров: {self.frame_count}")

def main():
    parser = argparse.ArgumentParser(
        description='CV-3: Подсчет и классификация объектов по цвету в реальном времени'
    )
    
    parser.add_argument('--source', type=str, default='0',
                       help='Источник видео')
    parser.add_argument('--min_area', type=int, default=500,
                       help='Минимальная площадь объекта в пикселях')
    parser.add_argument('--max_area', type=int, default=30000,
                       help='Максимальная площадь объекта в пикселях')
    parser.add_argument('--output', type=str, default='result.mp4',
                       help='Сохранить результат в видеофайл (опционально)')
    parser.add_argument('--stats', type=str, default='object_statistics.txt',
                       help='Имя файла для сохранения статистики')
    parser.add_argument('--resize', type=float, default=0.5,
                       help='Коэффициент изменения размера')
    
    args = parser.parse_args()
    
    counter = ObjectCounter(
        min_area=args.min_area,
        max_area=args.max_area,
    )
    
    try:
        if args.source == '0':
            cap = cv2.VideoCapture(0)
            print("Используется веб-камера")
        else:
            cap = cv2.VideoCapture(args.source)
            print(f"Используется видеофайл: {args.source}")
    except Exception as e:
        print(f"Ошибка при открытии видео источника: {e}")
        return
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео источник")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30
    
    print(f"\nПараметры видео:")
    print(f"  Разрешение: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    if total_frames > 0:
        print(f"  Всего кадров: {total_frames}")
    
    if args.resize != 1.0 and width * height > 1920 * 1080:
        new_width = int(width * args.resize)
        new_height = int(height * args.resize)
        print(f"  Изменение размера до: {new_width}x{new_height}")
    else:
        new_width = width
        new_height = height

    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps, 
            (new_width, new_height + 180)
        )
        print(f"Запись видео в файл: {args.output}")
    
    start_time = time.time()
    frame_index = 0

    window_name = 'CV-3-43'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 
                     min(1920, new_width), 
                     min(1080, new_height + 180))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Конец видео или ошибка чтения кадра")
                break
            
            frame_index += 1
            
            if args.resize != 1.0:
                frame = cv2.resize(frame, (new_width, new_height), 
                                    interpolation=cv2.INTER_AREA)
            
            processed_frame, _ = counter.process_frame(frame)    
            
            if video_writer:
                video_writer.write(processed_frame)
            
            cv2.imshow(window_name, processed_frame)
            cv2.waitKey(1)
                
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()
    finally:
        counter.save_statistics(args.stats)
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("ФИНАЛЬНЫЙ ОТЧЕТ")
        print("="*60)
        print(f"Всего обработано кадров: {counter.frame_count}")
        print(f"Общее время работы: {time.time() - start_time:.1f} секунд")
        if args.output:
            print(f"Видео сохранено в: {args.output}")
        print(f"Статистика сохранена в: {args.stats}")

if __name__ == "__main__":
    main()