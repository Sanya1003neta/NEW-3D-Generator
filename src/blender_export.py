"""
Blender Export Module
Експорт моделей в Blender та управління проектами
"""

import os
import json
import trimesh


class BlenderExporter:
    """Клас для експорту в Blender"""
    
    def __init__(self, model, car_info):
        """
        Ініціалізація експортера
        
        Args:
            model (trimesh.Trimesh): 3D модель
            car_info (dict): Інформація про автомобіль
        """
        self.model = model
        self.car_info = car_info
        
    def export_to_obj(self, output_path):
        """
        Експортує модель в OBJ формат
        
        Args:
            output_path (str): Шлях для збереження
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.model.export(output_path)
        print(f"✓ Модель експортована (OBJ): {output_path}")
    
    def export_to_ply(self, output_path):
        """
        Експортує модель в PLY формат
        
        Args:
            output_path (str): Шлях для збереження
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.model.export(output_path)
        print(f"✓ Модель експортована (PLY): {output_path}")
    
    def create_blender_project(self, project_dir):
        """
        Створює проект для Blender з моделлю та метаданами
        
        Args:
            project_dir (str): Директорія проекту
        """
        os.makedirs(project_dir, exist_ok=True)
        
        # Створюємо поддиректорії
        models_dir = os.path.join(project_dir, "models")
        textures_dir = os.path.join(project_dir, "textures")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(textures_dir, exist_ok=True)
        
        # Експортуємо модель
        obj_path = os.path.join(models_dir, "car_model.obj")
        self.export_to_obj(obj_path)
        
        # Зберігаємо метаінформацію
        self._save_metadata(project_dir)
        
        # Створюємо Blender Python скрипт
        self._create_blender_script(project_dir)
        
        print(f"✓ Проект Blender створений: {project_dir}")
        
        return project_dir
    
    def _save_metadata(self, project_dir):
        """Зберігає метаінформацію про модель"""
        metadata = {
            'car_type': self.car_info.get('car_type', 'unknown'),
            'dimensions': self.car_info.get('dimensions', {}),
            'model_info': {
                'vertices': len(self.model.vertices),
                'faces': len(self.model.faces),
                'volume': float(self.model.volume),
                'surface_area': float(self.model.area),
            },
            'export_info': self.car_info.get('features', {})
        }
        
        metadata_path = os.path.join(project_dir, "metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Метаінформація збережена: {metadata_path}")
    
    def _create_blender_script(self, project_dir):
        """Створює Python скрипт для Blender"""
        
        script_content = '''\"\"\"\nBlender Import Script для Car 3D Generator\nАвтоматично імпортує модель та налаштовує матеріали\n\"\"\"\n\nimport bpy\nimport os\nimport json\n\ndef import_car_model(project_dir):\n    \"\"\"Імпортує 3D модель автомобіля\"\"\"\n    \n    # Очищуємо сцену\n    bpy.ops.object.select_all(action='SELECT')\n    bpy.ops.object.delete()\n    \n    # Імпортуємо OBJ\n    obj_path = os.path.join(project_dir, "models", "car_model.obj")\n    \n    if os.path.exists(obj_path):\n        bpy.ops.import_scene.obj(filepath=obj_path)\n        print(f"✓ Модель імпортована: {obj_path}")\n    else:\n        print(f"⚠ Файл не знайдено: {obj_path}")\n        return\n    \n    # Завантажуємо метаданні\n    metadata_path = os.path.join(project_dir, "metadata.json")\n    \n    if os.path.exists(metadata_path):\n        with open(metadata_path, 'r', encoding='utf-8') as f:\n            metadata = json.load(f)\n        \n        print(f"Тип автомобіля: {metadata['car_type']}")\n        print(f"Розміри: {metadata['dimensions']}")\n    \n    # Застосовуємо матеріали\n    setup_materials()\n    \n    # Налаштовуємо освітлення\n    setup_lighting()\n    \n    print("✓ Сцена налаштована!")\n\ndef setup_materials():\n    \"\"\"Налаштовує матеріали для моделі\"\"\"\n    \n    # Отримуємо об'єкти\n    for obj in bpy.context.scene.objects:\n        if obj.type == 'MESH':\n            # Додаємо матеріал\n            mat = bpy.data.materials.new(name="CarMaterial")\n            mat.use_nodes = True\n            \n            # Налаштовуємо BSDF\n            bsdf = mat.node_tree.nodes["Principled BSDF"]\n            bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)\n            bsdf.inputs['Metallic'].default_value = 0.8\n            bsdf.inputs['Roughness'].default_value = 0.2\n            \n            # Додаємо матеріал до об'єкту\n            obj.data.materials.append(mat)\n    \n    print("✓ Матеріали застосовані")\n\ndef setup_lighting():\n    \"\"\"Налаштовує освітлення в сцені\"\"\"\n    \n    # Сонячне світло\n    sun_data = bpy.data.lights.new(name="Sun", type='SUN')\n    sun_data.energy = 2.0\n    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)\n    bpy.context.collection.objects.link(sun_obj)\n    sun_obj.location = (5, 5, 10)\n    \n    # Основне точкове світло\n    lamp_data = bpy.data.lights.new(name="MainLight", type='POINT')\n    lamp_data.energy = 1000\n    lamp_obj = bpy.data.objects.new(name="MainLight", object_data=lamp_data)\n    bpy.context.collection.objects.link(lamp_obj)\n    lamp_obj.location = (0, 0, 3)\n    \n    print("✓ Освітлення налаштоване")\n\n# Основна функція\nif __name__ == "__main__":\n    # Вказуємо шлях до проекту\n    project_dir = \"{project_dir}\"\n    import_car_model(project_dir)\n'''
        
        script_path = os.path.join(project_dir, "import_model.py")
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content.format(project_dir=project_dir))
        
        print(f"✓ Blender скрипт створений: {script_path}")
    
    def get_export_formats(self):
        """Повертає доступні формати експорту"""
        formats = {
            'obj': 'Wavefront OBJ (рекомендується для Blender)',
            'ply': 'Stanford PLY',
            'stl': 'STL (для 3D друку)',
            'collada': 'COLLADA DAE',
        }
        return formats


def export_for_blender(model, car_info, output_dir="blender_project"):
    """
    Основна функція для експорту в Blender
    
    Args:
        model (trimesh.Trimesh): 3D модель
        car_info (dict): Інформація про автомобіль
        output_dir (str): Директорія для проекту
    
    Returns:
        str: Шлях до проекту
    """
    exporter = BlenderExporter(model, car_info)
    project_path = exporter.create_blender_project(output_dir)
    
    print(f"""
✓ Проект готовий для Blender!

Для імпорту в Blender:
1. Відкрийте Blender
2. Перейдіть в Scripting режим
3. Відкрийте файл: {os.path.join(output_dir, 'import_model.py')}
4. Запустіть скрипт (Alt+P)

Або вручну імпортуйте: {os.path.join(output_dir, 'models', 'car_model.obj')}
    """)
    
    return project_path


if __name__ == "__main__":
    print("Blender Export Module")