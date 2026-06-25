"""
Advanced Model Optimization Module
Оптимізація 3D моделей для різних рівнів якості
"""

import numpy as np
import trimesh
import os


class ModelOptimizer:
    """Оптимізатор 3D моделей"""
    
    def __init__(self, mesh):
        """
        Ініціалізація оптимізатора
        
        Args:
            mesh (trimesh.Trimesh): 3D модель для оптимізації
        """
        self.original_mesh = mesh
        self.optimized_meshes = {}
        
        print(f"✓ Model Optimizer ініціалізований")
        print(f"  - Вершин: {len(mesh.vertices)}")
        print(f"  - Граней: {len(mesh.faces)}")
    
    def reduce_polygon_count(self, target_count, method='quadratic'):
        """
        Зменшує кількість полігонів моделі
        
        Args:
            target_count (int): Цільова кількість граней
            method (str): Метод зменшення ('quadratic', 'simplify')
        
        Returns:
            trimesh.Trimesh: Оптимізована модель
        """
        mesh = self.original_mesh.copy()
        
        print(f"\n📉 Зменшуємо полігони ({method})...")
        print(f"  Вихідно: {len(mesh.faces)} граней")
        
        # Використовуємо вбудовану функцію simplification
        try:
            mesh = mesh.simplify_quadratic_mesh(target_count=target_count)
        except:
            # Альтернативний метод
            ratio = target_count / len(mesh.faces)
            mesh = mesh.simplify_quadratic_mesh(target_reduction=1.0 - ratio)
        
        print(f"  Результат: {len(mesh.faces)} граней")
        print(f"  Зменшення: {(1 - len(mesh.faces) / len(self.original_mesh.faces)) * 100:.1f}%")
        
        self.optimized_meshes['reduced'] = mesh
        return mesh
    
    def create_lod_models(self, levels=[0.9, 0.5, 0.1]):
        """
        Створює моделі з різними рівнями деталізації (LOD)
        
        Args:
            levels (list): Список коефіцієнтів деталізації (0-1)
        
        Returns:
            dict: Словник LOD моделей
        """
        print(f"\n🔄 Створюємо LOD моделі ({len(levels)} рівнів)...")
        
        lod_models = {}
        original_faces = len(self.original_mesh.faces)
        
        for level in levels:
            target_faces = int(original_faces * level)
            print(f"\n  LOD Level {level}:")
            print(f"    Ціль: {target_faces} граней")
            
            try:
                lod_mesh = self.original_mesh.simplify_quadratic_mesh(
                    target_reduction=1.0 - level
                )
                lod_models[f'lod_{int(level*100)}'] = lod_mesh
                print(f"    ✓ Успішно: {len(lod_mesh.faces)} граней")
            except Exception as e:
                print(f"    ⚠ Помилка: {e}")
        
        self.optimized_meshes['lod'] = lod_models
        return lod_models
    
    def optimize_normals(self):
        """
        Оптимізує нормалі для краще освітлення
        """
        mesh = self.original_mesh.copy()
        
        # Перетворюємо в smooth shading
        mesh.smooth_shading = True
        
        # Об'єднуємо нормалі на спільних вершинах
        mesh.merge_vertices()
        
        print("✓ Нормалі оптимізовані (smooth shading активовано)")
        
        return mesh
    
    def create_bounding_volume_hierarchy(self):
        """
        Створює ієрархію обмежувальних об'ємів для швидкого перетину
        """
        mesh = self.original_mesh.copy()
        
        # Trimesh автоматично використовує BVH для перетинів
        bvh = mesh.bvh
        
        print(f"✓ BVH створена")
        print(f"  - Рівнів: {bvh.depth}")
        
        return bvh
    
    def export_optimized_models(self, output_dir='output/optimized'):
        """
        Експортує всі оптимізовані моделі
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 Експортуємо оптимізовані моделі...")
        
        # Експортуємо зменшену модель
        if 'reduced' in self.optimized_meshes:
            path = os.path.join(output_dir, 'car_optimized.obj')
            self.optimized_meshes['reduced'].export(path)
            print(f"  ✓ car_optimized.obj ({len(self.optimized_meshes['reduced'].faces)} граней)")
        
        # Експортуємо LOD моделі
        if 'lod' in self.optimized_meshes:
            for lod_name, lod_mesh in self.optimized_meshes['lod'].items():
                path = os.path.join(output_dir, f'car_{lod_name}.obj')
                lod_mesh.export(path)
                print(f"  ✓ car_{lod_name}.obj ({len(lod_mesh.faces)} граней)")
        
        print(f"\n✓ Все експортовано в: {output_dir}")
    
    def get_optimization_report(self):
        """
        Повертає звіт про оптимізацію
        """
        report = {
            'original': {
                'vertices': len(self.original_mesh.vertices),
                'faces': len(self.original_mesh.faces),
                'volume': self.original_mesh.volume,
                'surface_area': self.original_mesh.area,
            },
            'optimized': {}
        }
        
        for name, mesh in self.optimized_meshes.items():
            if isinstance(mesh, dict):  # LOD моделі
                report['optimized'][name] = {}
                for lod_name, lod_mesh in mesh.items():
                    report['optimized'][name][lod_name] = {
                        'faces': len(lod_mesh.faces),
                        'reduction': (1 - len(lod_mesh.faces) / len(self.original_mesh.faces)) * 100
                    }
            else:  # Одна модель
                report['optimized'][name] = {
                    'faces': len(mesh.faces),
                    'reduction': (1 - len(mesh.faces) / len(self.original_mesh.faces)) * 100
                }
        
        return report


class MeshValidator:
    """Валідатор 3D моделей для перевірки якості"""
    
    def __init__(self, mesh):
        """
        Ініціалізація валідатора
        
        Args:
            mesh (trimesh.Trimesh): Модель для перевірки
        """
        self.mesh = mesh
        self.issues = []
    
    def validate(self):
        """
        Проводить повну перевірку моделі
        """
        print("\n✓ Валідація моделі...")
        
        self.check_manifold()
        self.check_normals()
        self.check_bounds()
        self.check_watertight()
        self.check_degenerates()
        
        return self.get_report()
    
    def check_manifold(self):
        """Перевіряє чи модель є многовидом"""
        if self.mesh.is_watertight:
            print(f"  ✓ Модель замкнута (manifold)")
        else:
            print(f"  ⚠ Модель має отвори (non-manifold)")
            self.issues.append('non_manifold')
    
    def check_normals(self):
        """Перевіряє нормалі"""
        if len(self.mesh.vertex_normals) == len(self.mesh.vertices):
            print(f"  ✓ Нормалі коректні ({len(self.mesh.vertex_normals)} нормалей)")
        else:
            print(f"  ⚠ Помилка в нормалях")
            self.issues.append('bad_normals')
    
    def check_bounds(self):
        """Перевіряє межі моделі"""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        print(f"  ✓ Межі: {bounds.tolist()}")
        print(f"  ✓ Розмір: {size.tolist()}")
    
    def check_watertight(self):
        """Перевіряє чи модель герметична"""
        if self.mesh.is_watertight:
            print(f"  ✓ Модель герметична")
        else:
            print(f"  ⚠ Модель НЕ герметична")
            self.issues.append('not_watertight')
    
    def check_degenerates(self):
        """Перевіряє на вироджені грані"""
        area = self.mesh.area_faces
        degenerate = np.where(area < 1e-6)[0]
        
        if len(degenerate) == 0:
            print(f"  ✓ Немає виродженних граней")
        else:
            print(f"  ⚠ Виявлено {len(degenerate)} виродженних граней")
            self.issues.append('degenerate_faces')
    
    def get_report(self):
        """Повертає звіт валідації"""
        return {
            'is_valid': len(self.issues) == 0,
            'issues': self.issues,
            'is_watertight': self.mesh.is_watertight,
            'vertex_count': len(self.mesh.vertices),
            'face_count': len(self.mesh.faces),
        }


def optimize_model_advanced(mesh, reduction_target=0.3, create_lod=True):
    """
    Основна функція для розширеної оптимізації моделі
    
    Args:
        mesh (trimesh.Trimesh): Вихідна модель
        reduction_target (float): Цільовий рівень зменшення (0-1)
        create_lod (bool): Створювати LOD моделі
    
    Returns:
        dict: Результати оптимізації
    """
    print("\n🔧 РОЗШИРЕНА ОПТИМІЗАЦІЯ МОДЕЛІ")
    print("=" * 50)
    
    # Валідація
    validator = MeshValidator(mesh)
    validation_report = validator.validate()
    
    # Оптимізація
    optimizer = ModelOptimizer(mesh)
    
    # Зменшення полігонів
    target_faces = int(len(mesh.faces) * reduction_target)
    optimized = optimizer.reduce_polygon_count(target_faces)
    
    # Оптимізація нормалей
    optimizer.optimize_normals()
    
    # LOD моделі
    lod_models = None
    if create_lod:
        lod_models = optimizer.create_lod_models(levels=[0.9, 0.7, 0.5, 0.3, 0.1])
    
    # Звіт
    report = optimizer.get_optimization_report()
    
    print("\n📊 ЗВІТ ОПТИМІЗАЦІЇ:")
    print(f"  Вихідна модель:")
    print(f"    - Грані: {report['original']['faces']}")
    print(f"    - Площа: {report['original']['surface_area']:.2f}")
    
    return {
        'optimized_mesh': optimized,
        'lod_models': lod_models,
        'validation': validation_report,
        'report': report
    }


if __name__ == "__main__":
    print("Advanced Model Optimization Module")
