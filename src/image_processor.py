"""
Image Processing Module
Обробка фотографій автомобілів та виділення контурів
"""

import cv2
import numpy as np
from PIL import Image
import os


class ImageProcessor:
    """Клас для обробки фотографій автомобілів"""
    
    def __init__(self, image_path):
        """
        Ініціалізація процесора
        
        Args:
            image_path (str): Шлях до фотографії
        """
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.contours = None
        self.edges = None
        
    def load_image(self):
        """Завантажує зображення"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Файл не знайдено: {self.image_path}")
        
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError("Не вдалось завантажити зображення")
        
        print(f"✓ Зображення завантажено: {self.image_path}")
        return self.original_image
    
    def resize_image(self, width=800, height=600):
        """Змінює розмір зображення"""
        self.original_image = cv2.resize(self.original_image, (width, height))
        print(f"✓ Розмір змінено на {width}x{height}")
        return self.original_image
    
    def detect_edges(self):
        """Виділяє контури за допомогою Canny edge detection"""
        # Конвертуємо в сірий
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Застосовуємо розмиття
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        self.edges = cv2.Canny(blurred, 50, 150)
        
        print("✓ Контури виділені (Canny)")
        return self.edges
    
    def find_contours(self):
        """Знаходить контури на зображенні"""
        contours, _ = cv2.findContours(
            self.edges, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Сортуємо контури за площею
        self.contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        print(f"✓ Знайдено контурів: {len(self.contours)}")
        return self.contours
    
    def get_car_silhouette(self):
        """Отримує силует автомобіля (найбільший контур)"""
        if not self.contours:
            self.detect_edges()
            self.find_contours()
        
        # Беремо найбільший контур (це мають бути автомобіль)
        car_contour = self.contours[0] if self.contours else None
        
        if car_contour is None:
            print("⚠ Контур автомобіля не знайдено")
            return None
        
        # Апроксимуємо контур
        epsilon = 0.02 * cv2.arcLength(car_contour, True)
        approx = cv2.approxPolyDP(car_contour, epsilon, True)
        
        print(f"✓ Силует отримано ({len(approx)} точок)")
        return approx
    
    def get_bounding_box(self):
        """Отримує обмежувальний прямокутник автомобіля"""
        silhouette = self.get_car_silhouette()
        
        if silhouette is None:
            return None
        
        x, y, w, h = cv2.boundingRect(silhouette)
        
        print(f"✓ Bounding box: x={x}, y={y}, w={w}, h={h}")
        return (x, y, w, h)
    
    def extract_features(self):
        """Виділяє ключові ознаки автомобіля"""
        silhouette = self.get_car_silhouette()
        
        if silhouette is None:
            return None
        
        features = {
            'contour_points': len(silhouette),
            'area': cv2.contourArea(silhouette),
            'perimeter': cv2.arcLength(silhouette, True),
            'bounding_box': cv2.boundingRect(silhouette),
            'moments': cv2.moments(silhouette),
        }
        
        # Цент маси
        if features['moments']['m00'] != 0:
            cx = int(features['moments']['m10'] / features['moments']['m00'])
            cy = int(features['moments']['m01'] / features['moments']['m00'])
            features['center'] = (cx, cy)
        
        print(f"✓ Ознаки виділені:")
        print(f"  - Площа: {features['area']:.0f}")
        print(f"  - Периметр: {features['perimeter']:.0f}")
        
        return features
    
    def save_processed(self, output_path):
        """Зберігає оброблене зображення"""
        if self.original_image is None:
            print("⚠ Зображення не оброблено")
            return
        
        cv2.imwrite(output_path, self.original_image)
        print(f"✓ Зображення збережено: {output_path}")
    
    def draw_contours(self, output_path):
        """Малює контури на зображенні та зберігає"""
        result = self.original_image.copy()
        cv2.drawContours(result, self.contours[:5], -1, (0, 255, 0), 2)
        cv2.imwrite(output_path, result)
        print(f"✓ Контури намальовані: {output_path}")


def process_car_image(image_path, output_dir="output"):
    """
    Основна функція для обробки фотографії автомобіля
    
    Args:
        image_path (str): Шлях до фото
        output_dir (str): Директорія для результатів
    
    Returns:
        dict: Ознаки автомобіля
    """
    processor = ImageProcessor(image_path)
    
    # Завантажуємо
    processor.load_image()
    processor.resize_image(800, 600)
    
    # Виділяємо контури
    processor.detect_edges()
    processor.find_contours()
    
    # Отримуємо ознаки
    features = processor.extract_features()
    
    # Зберігаємо результати
    os.makedirs(output_dir, exist_ok=True)
    processor.draw_contours(f"{output_dir}/contours.jpg")
    processor.save_processed(f"{output_dir}/processed.jpg")
    
    return features


if __name__ == "__main__":
    # Приклад використання
    print("Image Processor Module")