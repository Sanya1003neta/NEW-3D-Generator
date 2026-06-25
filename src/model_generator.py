"""
3D Model Generator Module
Генерація 3D моделей з контурів та параметрів автомобіля
"""

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import os


class ModelGenerator:
    """Клас для генерації 3D моделей автомобілів"""
    
    def __init__(self, features, image_shape=(600, 800)):
        """
        Ініціалізація генератора
        
        Args:
            features (dict): Ознаки автомобіля від ImageProcessor
            image_shape (tuple): Розміри зображення (height, width)
        """
        self.features = features
        self.image_shape = image_shape
        self.mesh = None
        self.vertices = None
        self.faces = None
        
    def generate_basic_car_mesh(self):
        """Генерує базову 3D модель автомобіля"""
        
        # Отримуємо розміри
        bbox = self.features.get('bounding_box', (0, 0, 400, 300))
        x, y, w, h = bbox
        
        # Масштаб для 3D
        scale = 0.01
        
        # Створюємо базові вершини (8 точок куба)
        length = w * scale  # Довжина
        width = 1.5 * scale  # Ширина
        height = 1.0 * scale  # Висота
        
        # Вершини основного тіла машини
        base_vertices = np.array([
            # Передня частина
            [-length/2, -width/2, 0],  # 0
            [length/2, -width/2, 0],   # 1
            [length/2, width/2, 0],    # 2
            [-length/2, width/2, 0],   # 3
            # Задня частина (вище)
            [-length/2, -width/2, height],  # 4
            [length/2, -width/2, height],   # 5
            [length/2, width/2, height],    # 6
            [-length/2, width/2, height],   # 7
        ])
        
        # Додаємо точки для капота (нахилену форму)
        hood_vertices = np.array([
            [-length/4, -width/2, height*0.5],  # 8
            [length/4, -width/2, height*0.5],   # 9
        ])
        
        self.vertices = np.vstack([base_vertices, hood_vertices])
        
        # Визначаємо грані (faces)
        self.faces = np.array([
            # Нижня грань
            [0, 1, 2], [0, 2, 3],
            # Верхня грань
            [4, 6, 5], [4, 7, 6],
            # Передня грань
            [0, 4, 5], [0, 5, 1],
            # Задня грань
            [2, 6, 7], [2, 7, 3],
            # Ліва грань
            [0, 3, 7], [0, 7, 4],
            # Права грань
            [1, 5, 6], [1, 6, 2],
        ])
        
        # Створюємо mesh
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        
        print(f"✓ Базова модель створена")
        print(f"  - Вершин: {len(self.vertices)}")
        print(f"  - Граней: {len(self.faces)}")
        
        return self.mesh
    
    def generate_advanced_mesh(self):
        """Генерує більш складну модель з деталями"""
        
        if self.mesh is None:
            self.generate_basic_car_mesh()
        
        # Додаємо колеса (циліндри)
        wheel_vertices = []
        wheel_faces = []
        
        bbox = self.features.get('bounding_box', (0, 0, 400, 300))
        x, y, w, h = bbox
        scale = 0.01
        
        # Радіус колеса
        wheel_radius = 0.3 * scale
        height_wheel = 0.2 * scale
        
        # Позиції коліс
        wheel_positions = [
            (-w*scale/3, -1.5*scale, 0),  # Передній лівий
            (w*scale/3, -1.5*scale, 0),   # Передній правий
            (-w*scale/3, 1.5*scale, 0),   # Задній лівий
            (w*scale/3, 1.5*scale, 0),    # Задній правий
        ]
        
        for i, (cx, cy, cz) in enumerate(wheel_positions):
            # Створюємо циліндр для колеса
            wheel = trimesh.primitives.Cylinder(
                radius=wheel_radius,
                height=height_wheel,
                sections=16
            )
            wheel.apply_translation([cx, cy, cz])
            
            if i == 0:
                wheels_mesh = wheel
            else:
                wheels_mesh = trimesh.util.concatenate([wheels_mesh, wheel])
        
        # Об'єднуємо основну модель з колесами
        self.mesh = trimesh.util.concatenate([self.mesh, wheels_mesh])
        
        print("✓ Деталі (колеса) додані до моделі")
        
        return self.mesh
    
    def smooth_mesh(self, iterations=2):
        """Згладжує mesh для більш натурального виду"""
        if self.mesh is None:
            self.generate_advanced_mesh()
        
        # Застосовуємо Laplacian smoothing
        for _ in range(iterations):
            self.mesh.smooth_shading()
        
        print(f"✓ Mesh згладжено ({iterations} ітерацій)")
        return self.mesh
    
    def export_obj(self, output_path):
        """Експортує модель в OBJ формат"""
        if self.mesh is None:
            self.generate_advanced_mesh()
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        self.mesh.export(output_path)
        print(f"✓ Модель експортована (OBJ): {output_path}")
    
    def export_fbx(self, output_path):
        """Експортує модель в FBX формат"""
        if self.mesh is None:
            self.generate_advanced_mesh()
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Примітка: trimesh не має вбудованої підтримки FBX,
        # але можемо експортувати в OBJ та використати Blender
        self.mesh.export(output_path.replace('.fbx', '.obj'))
        print(f"✓ Модель експортована (OBJ, можна конвертувати в FBX): {output_path.replace('.fbx', '.obj')}")
    
    def get_model_info(self):
        """Повертає інформацію про модель"""
        if self.mesh is None:
            return None
        
        info = {
            'vertices': len(self.mesh.vertices),
            'faces': len(self.mesh.faces),
            'volume': self.mesh.volume,
            'surface_area': self.mesh.area,
            'bounds': self.mesh.bounds,
            'center_of_mass': self.mesh.center_of_mass,
        }
        
        return info


def generate_car_model(features, image_shape=(600, 800), output_obj=None):
    """
    Основна функція для генерації 3D моделі
    
    Args:
        features (dict): Ознаки від ImageProcessor
        image_shape (tuple): Розміри зображення
        output_obj (str): Шлях для експорту OBJ
    
    Returns:
        trimesh.Trimesh: Згенерована 3D модель
    """
    generator = ModelGenerator(features, image_shape)
    
    # Генеруємо модель
    generator.generate_advanced_mesh()
    generator.smooth_mesh()
    
    # Експортуємо якщо вказано
    if output_obj:
        generator.export_obj(output_obj)
    
    # Виводимо інформацію
    info = generator.get_model_info()
    print("\n📊 Інформація про модель:")
    print(f"  - Вершин: {info['vertices']}")
    print(f"  - Граней: {info['faces']}")
    print(f"  - Об'єм: {info['volume']:.4f}")
    print(f"  - Площа поверхні: {info['surface_area']:.4f}")
    
    return generator.mesh


if __name__ == "__main__":
    print("Model Generator Module")