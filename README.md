# 🚗 Car 3D Generator

Програма для автоматичної генерації 3D моделей автомобілів з фотографій за допомогою машинного навчання та експорту в Blender.

## 📋 Функціональність

### Рівень 1: Середній (поточний)
- ✅ Аналіз фотографії автомобіля
- ✅ Виділення контурів та форми
- ✅ Генерація базової 3D сітки
- ✅ Експорт в Blender (.obj, .fbx)
- ✅ Застосування текстур

### Рівень 2: Складний (планується)
- 🔄 Глибока нейросеть (PyTorch/TensorFlow)
- 🔄 ML модель для розпізнавання деталей авто
- 🔄 Генерація високоякісних 3D моделей
- 🔄 Текстури з ІЇ

## 🛠️ Вимоги

```
Python 3.8+
OpenCV
NumPy
Pillow
bpy (Blender)
scikit-learn
Trimesh
```

## 📦 Встановлення

```bash
git clone https://github.com/Sanya1003neta/NEW-3D-Generator.git
cd NEW-3D-Generator
pip install -r requirements.txt
```

## 🚀 Використання

```bash
python main.py --image path/to/car_photo.jpg --output model.obj
```

## 📁 Структура проекту

```
NEW-3D-Generator/
├── main.py                 # Головна програма
├── requirements.txt        # Залежності
├── README.md              # Документація
├── src/
│   ├── image_processor.py  # Обробка фотографій
│   ├── model_generator.py  # Генерація 3D моделі
│   ├── ml_analyzer.py      # ML аналіз автомобіля
│   └── blender_export.py   # Експорт в Blender
├── models/                # Навчені ML моделі
├── input/                 # Вхідні фотографії
└── output/               # Готові 3D моделі
```

## 📸 Приклад роботи

1. Завантажте фото автомобіля
2. Програма аналізує форму та контури
3. Генерує 3D модель
4. Експортує в Blender

## 📝 Ліцензія

MIT License - див. LICENSE файл

## 👨‍💻 Автор

Sanya1003neta

---

**Рівень складності:** 🟨 Середній → 🟥 Складний (в розробці)