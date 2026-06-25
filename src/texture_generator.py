"""
Texture Generation Module - Advanced Level
Генерація текстур з ІІ для автомобілів
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import json

try:
    import torch
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class TextureGenerator:
    """Генератор текстур для 3D моделей"""
    
    def __init__(self, texture_size=512):
        """
        Ініціалізація генератора текстур
        
        Args:
            texture_size (int): Розмір текстури (512x512, 1024x1024 тощо)
        """
        self.texture_size = texture_size
        self.device = 'cuda' if PYTORCH_AVAILABLE else 'cpu'
        self.pipeline = None
        
        print(f"✓ Texture Generator ініціалізований")
        print(f"  - Розмір текстури: {texture_size}x{texture_size}")
        print(f"  - Device: {self.device}")
    
    def generate_car_paint(self, color=(0.8, 0.8, 0.8), roughness=0.3):
        """
        Генерує текстуру фарби автомобіля
        
        Args:
            color (tuple): RGB колір (0-1)
            roughness (float): Шорсткість поверхні
        
        Returns:
            PIL.Image: Текстура фарби
        """
        # Створюємо базовий колір
        base_texture = Image.new('RGB', (self.texture_size, self.texture_size))
        pixels = base_texture.load()
        
        # Конвертуємо колір в 0-255 діапазон
        r, g, b = [int(c * 255) for c in color]
        
        # Додаємо текстуру з шумом
        noise = np.random.randint(0, 50, (self.texture_size, self.texture_size, 3))
        
        for i in range(self.texture_size):
            for j in range(self.texture_size):
                nr = max(0, min(255, r + noise[i, j, 0]))
                ng = max(0, min(255, g + noise[i, j, 1]))
                nb = max(0, min(255, b + noise[i, j, 2]))
                pixels[i, j] = (nr, ng, nb)
        
        # Застосовуємо фільтри для реалістичності
        texture = base_texture.filter(ImageFilter.GaussianBlur(radius=2))
        texture = texture.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        print(f"✓ Текстура фарби створена: RGB{color}")
        return texture
    
    def generate_metallic_texture(self, color=(0.5, 0.5, 0.5), metallic=0.9):
        """Генерує металічну текстуру"""
        base = self.generate_car_paint(color=color)
        
        # Додаємо металічний ефект
        arr = np.array(base).astype(float) / 255.0
        
        # Посилюємо світлі області для металічного ефекту
        arr = arr ** (1.0 / (metallic + 0.1))
        arr = (arr * 255).astype(np.uint8)
        
        metallic_texture = Image.fromarray(arr)
        
        print(f"✓ Металічна текстура створена (metallic={metallic})")
        return metallic_texture
    
    def generate_normal_map(self, base_texture):
        """Генерує карту нормалей з базової текстури"""
        # Конвертуємо в greyscale
        gray = base_texture.convert('L')
        arr = np.array(gray).astype(float) / 255.0
        
        # Розраховуємо градієнти
        gx = np.gradient(arr, axis=1)
        gy = np.gradient(arr, axis=0)
        
        # Нормалізуємо
        normal = np.zeros((self.texture_size, self.texture_size, 3))
        normal[:, :, 0] = gx  # X (Red)
        normal[:, :, 1] = gy  # Y (Green)
        normal[:, :, 2] = np.ones_like(gx)  # Z (Blue)
        
        # Нормалізуємо вектори
        magnitude = np.sqrt(normal[:, :, 0]**2 + normal[:, :, 1]**2 + normal[:, :, 2]**2)
        magnitude[magnitude == 0] = 1
        normal = normal / magnitude[:, :, np.newaxis]
        
        # Конвертуємо в 0-255
        normal = ((normal + 1) * 127.5).astype(np.uint8)
        normal_map = Image.fromarray(normal)
        
        print(f"✓ Карта нормалей створена")
        return normal_map
    
    def generate_roughness_map(self, base_texture):
        """Генерує карту шорсткості"""
        gray = base_texture.convert('L')
        arr = np.array(gray).astype(np.uint8)
        
        # Інвертуємо (білі області = гладкі)
        roughness_map = ImageOps.invert(gray)
        
        print(f"✓ Карта шорсткості створена")
        return roughness_map
    
    def generate_metallic_map(self, pattern='uniform'):
        """Генерує карту металічності"""
        if pattern == 'uniform':
            # Рівномірна металічність
            metallic_map = Image.new('L', (self.texture_size, self.texture_size), 200)
        
        elif pattern == 'edges':
            # Металічність на краях
            metallic_map = Image.new('L', (self.texture_size, self.texture_size), 100)
            draw = ImageDraw.Draw(metallic_map)
            draw.rectangle([0, 0, self.texture_size, 20], fill=255)  # Верх
            draw.rectangle([0, self.texture_size-20, self.texture_size, self.texture_size], fill=255)  # Низ
        
        elif pattern == 'accents':
            # Металічні деталі
            metallic_map = Image.new('L', (self.texture_size, self.texture_size), 50)
            draw = ImageDraw.Draw(metallic_map)
            # Деталі вікон
            draw.rectangle([50, 100, 150, 200], fill=200)
            draw.rectangle([self.texture_size-150, 100, self.texture_size-50, 200], fill=200)
        
        else:
            metallic_map = Image.new('L', (self.texture_size, self.texture_size), 150)
        
        print(f"✓ Карта металічності створена (pattern={pattern})")
        return metallic_map
    
    def generate_complete_pbr_textures(self, car_color=(0.2, 0.4, 0.8)):
        """Генерує повний набір PBR текстур"""
        print(f"\n🎨 Генеруємо повний набір PBR текстур...")
        
        base_texture = self.generate_car_paint(color=car_color)
        
        textures = {
            'albedo': base_texture,
            'normal': self.generate_normal_map(base_texture),
            'roughness': self.generate_roughness_map(base_texture),
            'metallic': self.generate_metallic_map(pattern='uniform'),
            'metallic_accents': self.generate_metallic_map(pattern='accents'),
        }
        
        print(f"✓ Повний набір PBR текстур готовий ({len(textures)} текстур)")
        return textures
    
    def save_textures(self, textures, output_dir='textures'):
        """Зберігає текстури в файли"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, texture in textures.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            texture.save(filepath)
            print(f"  ✓ {name}: {filepath}")
        
        # Зберігаємо метаданні
        metadata = {
            'texture_size': self.texture_size,
            'textures': list(textures.keys()),
            'format': 'PNG',
            'pbr_ready': True
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Текстури збережені в: {output_dir}")
    
    def preview_textures(self, textures, output_path='textures_preview.png'):
        """Створює превью всіх текстур"""
        # Створюємо сітку текстур
        cols = 2
        rows = (len(textures) + cols - 1) // cols
        
        preview_size = self.texture_size // 2
        preview = Image.new('RGB', (cols * preview_size, rows * preview_size), (255, 255, 255))
        
        for idx, (name, texture) in enumerate(textures.items()):
            # Зменшуємо текстуру для превью
            resized = texture.convert('RGB').resize((preview_size, preview_size))
            
            # Визначаємо позицію
            row = idx // cols
            col = idx % cols
            x = col * preview_size
            y = row * preview_size
            
            preview.paste(resized, (x, y))
        
        preview.save(output_path)
        print(f"✓ Превью текстур створено: {output_path}")
        
        return preview


class AITextureEnhancer:
    """Покращувач текстур з використанням ІІ"""
    
    def __init__(self):
        """Ініціалізація покращувача"""
        self.model = None
        self.device = 'cuda' if PYTORCH_AVAILABLE else 'cpu'
        
        if DIFFUSERS_AVAILABLE:
            print("✓ Diffusers доступні")
        else:
            print("⚠ Diffusers не встановлені. Встановіть: pip install diffusers")
    
    def enhance_texture_resolution(self, texture, scale_factor=2):
        """Покращує розділення текстури (upscaling)"""
        new_size = (texture.width * scale_factor, texture.height * scale_factor)
        upscaled = texture.resize(new_size, Image.Resampling.LANCZOS)
        
        print(f"✓ Текстура масштабована: {texture.size} → {upscaled.size}")
        return upscaled
    
    def enhance_details(self, texture, enhancement_strength=1.5):
        """Посилює деталі текстури"""
        # Застосовуємо unsharpen mask для посилення деталей
        enhanced = texture.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=int(150 * enhancement_strength),
            threshold=3
        ))
        
        print(f"✓ Деталі текстури посилені (strength={enhancement_strength})")
        return enhanced


def generate_complete_car_textures(car_color=(0.2, 0.4, 0.8), output_dir='output/textures'):
    """
    Основна функція для генерації всіх текстур автомобіля
    
    Args:
        car_color (tuple): RGB колір автомобіля
        output_dir (str): Директорія для збереження
    
    Returns:
        dict: Словник усіх текстур
    """
    print("\n🎨 Генеруємо текстури для автомобіля...")
    
    # Генеруємо базові текстури
    generator = TextureGenerator(texture_size=512)
    textures = generator.generate_complete_pbr_textures(car_color=car_color)
    
    # Покращуємо текстури
    enhancer = AITextureEnhancer()
    textures['albedo'] = enhancer.enhance_details(textures['albedo'])
    
    # Зберігаємо
    generator.save_textures(textures, output_dir=output_dir)
    generator.preview_textures(textures, output_path=os.path.join(output_dir, 'preview.png'))
    
    return textures


if __name__ == "__main__":
    print("Texture Generation Module (Advanced Level)")
