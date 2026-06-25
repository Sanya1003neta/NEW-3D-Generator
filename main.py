#!/usr/bin/env python3
"""
Car 3D Generator - Main Program
Програма для генерації 3D моделей автомобілів з фотографій
"""

import os
import sys
import argparse
from pathlib import Path

# Додаємо src директорію до path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.image_processor import process_car_image
from src.model_generator import generate_car_model
from src.ml_analyzer import analyze_car
from src.blender_export import export_for_blender


def main():
    """Основна функція програми"""
    
    parser = argparse.ArgumentParser(
        description='🚗 Car 3D Generator - Генерація 3D моделей автомобілів з фото',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  python main.py --image photo.jpg
  python main.py --image photo.jpg --output my_car.obj
  python main.py --image photo.jpg --level advanced
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Шлях до фотографії автомобіля'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output/car_model.obj',
        help='Шлях для збереження 3D моделі (default: output/car_model.obj)'
    )
    
    parser.add_argument(
        '--level', '-l',
        type=str,
        choices=['basic', 'medium', 'advanced'],
        default='medium',
        help='Рівень деталізації (basic/medium/advanced, default: medium)'
    )
    
    parser.add_argument(
        '--blender', '-b',
        type=str,
        nargs='?',
        const='blender_project',
        help='Створити проект для Blender (опціональний шлях)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Детальний вивід інформації'
    )
    
    args = parser.parse_args()
    
    # Виводимо приголошення
    print_header()
    
    try:
        # Крок 1: Обробка зображення
        print("\n🖼️  Крок 1: Обробка фотографії...")
        print("-" * 50)
        
        if not os.path.exists(args.image):
            print(f"❌ Помилка: Файл не знайдено: {args.image}")
            sys.exit(1)
        
        features = process_car_image(args.image, output_dir="output")
        
        # Крок 2: ML аналіз
        print("\n🤖 Крок 2: ML аналіз автомобіля...")
        print("-" * 50)
        
        analysis = analyze_car(features)
        
        if args.verbose:
            print(f"\nТип: {analysis['car_type']}")
            print(f"Розміри: {analysis['dimensions']}")
        
        # Крок 3: Генерація 3D моделі
        print("\n🔨 Крок 3: Генерація 3D моделі...")
        print("-" * 50)
        
        model = generate_car_model(features, output_obj=args.output)
        
        # Крок 4: Експорт (опціонально)
        if args.blender:
            print("\n📦 Крок 4: Експорт для Blender...")
            print("-" * 50)
            
            export_for_blender(model, analysis, output_dir=args.blender)
        
        # Фінальне повідомлення
        print("\n" + "=" * 50)
        print("✅ УСПІШНО! Модель створена!")
        print("=" * 50)
        print(f"\n📁 Вихідний файл: {os.path.abspath(args.output)}")
        
        if args.blender:
            print(f"📦 Проект Blender: {os.path.abspath(args.blender)}")
        
        print(f"\n📊 Статистика:")
        print(f"   - Тип автомобіля: {analysis['car_type']}")
        print(f"   - Розміри: {analysis['dimensions']}")
        
    except Exception as e:
        print(f"\n❌ Помилка: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_header():
    """Виводить заголовок програми"""
    header = """
╔══════════════════════════════════════════════════╗
║         🚗 CAR 3D GENERATOR 🚗                   ║
║   Генерація 3D моделей автомобілів з фото       ║
║                                                  ║
║   Рівень: СЕРЕДНІЙ (Medium)                     ║
║   Версія: 1.0.0                                 ║
╚══════════════════════════════════════════════════╝
    """
    print(header)


if __name__ == "__main__":
    main()