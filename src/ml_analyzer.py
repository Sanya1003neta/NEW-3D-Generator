"""
ML Analyzer Module
Машинне навчання для аналізу та класифікації автомобілів
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage import measure
import pickle
import os


class CarAnalyzer:
    """Клас для ML аналізу автомобілів"""
    
    def __init__(self):
        """Ініціалізація аналізатора"""
        self.model = None
        self.scaler = StandardScaler()
        self.car_types = ['sedan', 'suv', 'truck', 'sports', 'hatchback']
        
    def extract_ml_features(self, features):
        """
        Виділяє ознаки для ML моделі
        
        Args:
            features (dict): Ознаки від ImageProcessor
        
        Returns:
            np.array: Вектор ознак
        """
        bbox = features.get('bounding_box', (0, 0, 400, 300))
        x, y, w, h = bbox
        
        area = features.get('area', 0)
        perimeter = features.get('perimeter', 0)
        
        # Розраховуємо додаткові ознаки
        aspect_ratio = w / (h + 0.0001)  # Ширина/Висота
        compactness = (perimeter ** 2) / (area + 0.0001)  # Компактність
        elongation = w / (h + 0.0001)  # Видовженість
        
        # Вектор ознак для ML
        ml_features = np.array([
            area,
            perimeter,
            aspect_ratio,
            compactness,
            elongation,
            w,
            h,
            w * h,  # Площа bbox
        ])
        
        return ml_features
    
    def train_classifier(self, training_data, labels):
        """
        Тренує класифікатор
        
        Args:
            training_data (np.array): Матриця ознак (n_samples, n_features)
            labels (np.array): Мітки класів
        """
        # Нормалізуємо дані
        X_scaled = self.scaler.fit_transform(training_data)
        
        # Тренуємо Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.model.fit(X_scaled, labels)
        
        print("✓ Класифікатор протренований")
        print(f"  - Типи автомобілів: {self.car_types}")
        print(f"  - N_estimators: 100")
    
    def predict_car_type(self, features):
        """
        Передбачає тип автомобіля
        
        Args:
            features (dict): Ознаки від ImageProcessor
        
        Returns:
            str: Тип автомобіля
        """
        if self.model is None:
            print("⚠ Модель не протренована. Виконуємо базовий аналіз...")
            return self.basic_car_type_analysis(features)
        
        # Виділяємо ознаки
        ml_features = self.extract_ml_features(features)
        ml_features = ml_features.reshape(1, -1)
        
        # Нормалізуємо
        ml_features_scaled = self.scaler.transform(ml_features)
        
        # Передбачаємо
        prediction = self.model.predict(ml_features_scaled)[0]
        probability = self.model.predict_proba(ml_features_scaled)[0]
        
        print(f"✓ Тип автомобіля: {prediction}")
        print(f"  - Впевненість: {max(probability)*100:.1f}%")
        
        return prediction
    
    def basic_car_type_analysis(self, features):
        """
        Базовий аналіз типу автомобіля без ML
        """
        bbox = features.get('bounding_box', (0, 0, 400, 300))
        x, y, w, h = bbox
        
        aspect_ratio = w / (h + 0.0001)
        
        # Проста евристика
        if aspect_ratio > 2.0:
            car_type = "truck"  # Вантажівка широка та низька
        elif aspect_ratio > 1.6:
            car_type = "suv"    # SUV середнього пропорційного
        elif aspect_ratio > 1.3:
            car_type = "sedan"  # Седан звичайний
        elif aspect_ratio > 0.9:
            car_type = "hatchback"  # Хетчбек компактний
        else:
            car_type = "sports"  # Спортсар високий
        
        print(f"✓ Тип автомобіля (базовий аналіз): {car_type}")
        print(f"  - Aspect ratio: {aspect_ratio:.2f}")
        
        return car_type
    
    def estimate_dimensions(self, features, car_type):
        """
        Оцінює розміри автомобіля на основі типу
        
        Args:
            features (dict): Ознаки від ImageProcessor
            car_type (str): Тип автомобіля
        
        Returns:
            dict: Орієнтовні розміри
        """
        bbox = features.get('bounding_box', (0, 0, 400, 300))
        x, y, w, h = bbox
        
        # Типові розміри автомобілів (метри)
        typical_dimensions = {
            'sedan': {'length': 4.7, 'width': 1.8, 'height': 1.5},
            'suv': {'length': 4.8, 'width': 1.9, 'height': 1.7},
            'truck': {'length': 5.5, 'width': 2.0, 'height': 1.8},
            'sports': {'length': 4.5, 'width': 1.9, 'height': 1.3},
            'hatchback': {'length': 4.2, 'width': 1.8, 'height': 1.5},
        }
        
        dims = typical_dimensions.get(car_type, typical_dimensions['sedan'])
        
        # Коригуємо за аспект-рейшеном
        aspect_ratio = w / (h + 0.0001)
        dims['length'] *= aspect_ratio / 2.6
        
        print(f"✓ Орієнтовні розміри для {car_type}:")
        print(f"  - Довжина: {dims['length']:.2f}м")
        print(f"  - Ширина: {dims['width']:.2f}м")
        print(f"  - Висота: {dims['height']:.2f}м")
        
        return dims
    
    def save_model(self, filepath):
        """Зберігає натреновану модель"""
        if self.model is None:
            print("⚠ Немає моделі для збереження")
            return
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'car_types': self.car_types
        }
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Модель збережена: {filepath}")
    
    def load_model(self, filepath):
        """Завантажує натреновану модель"""
        if not os.path.exists(filepath):
            print(f"⚠ Модель не знайдена: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.car_types = data['car_types']
        
        print(f"✓ Модель завантажена: {filepath}")
        return True


def analyze_car(features):
    """
    Основна функція для аналізу автомобіля
    
    Args:
        features (dict): Ознаки від ImageProcessor
    
    Returns:
        dict: Результати аналізу
    """
    analyzer = CarAnalyzer()
    
    # Передбачаємо тип
    car_type = analyzer.predict_car_type(features)
    
    # Оцінюємо розміри
    dimensions = analyzer.estimate_dimensions(features, car_type)
    
    return {
        'car_type': car_type,
        'dimensions': dimensions,
        'features': features
    }


if __name__ == "__main__":
    print("ML Analyzer Module")