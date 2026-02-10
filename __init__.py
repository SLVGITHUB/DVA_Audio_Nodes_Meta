"""
Audio Nodes for ComfyUI
Набор нод для анализа и обработки аудио файлов
Автор: [Ваше имя/компания]
Версия: 1.0.0
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DVA_Audio_Nodes")

# Проверка Python версии
PYTHON_VERSION = sys.version_info
if PYTHON_VERSION < (3, 8):
    logger.error(f"❌ Требуется Python 3.8 или выше. Текущая версия: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}")
    raise RuntimeError(f"Несовместимая версия Python: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}")

# Проверка наличия ComfyUI
try:
    import folder_paths
    import comfy.utils
    COMFYUI_AVAILABLE = True
    logger.info("✅ ComfyUI обнаружен")
except ImportError:
    COMFYUI_AVAILABLE = False
    logger.error("❌ ComfyUI не обнаружен. Убедитесь, что ноды установлены в правильную директорию.")
    raise

# Проверка и импорт зависимостей
DEPENDENCIES = {
    "pydub": False,
    "librosa": False,
    "soundfile": False,
    "numpy": False,
    "ffmpeg": False
}

try:
    import numpy as np
    DEPENDENCIES["numpy"] = True
    logger.info("✅ NumPy обнаружен")
except ImportError:
    logger.warning("⚠️  NumPy не установлен. Установите: pip install numpy")

try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    DEPENDENCIES["pydub"] = True
    logger.info("✅ PyDub обнаружен")
except ImportError:
    logger.warning("⚠️  PyDub не установлен. Некоторые функции могут быть недоступны. Установите: pip install pydub")

try:
    import librosa
    DEPENDENCIES["librosa"] = True
    logger.info("✅ Librosa обнаружен")
except ImportError:
    logger.warning("⚠️  Librosa не установлен. Некоторые функции могут быть недоступны. Установите: pip install librosa")

try:
    import soundfile as sf
    DEPENDENCIES["soundfile"] = True
    logger.info("✅ SoundFile обнаружен")
except ImportError:
    logger.warning("⚠️  SoundFile не установлен. Некоторые функции могут быть недоступны. Установите: pip install soundfile")

# Проверка ffmpeg
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        DEPENDENCIES["ffmpeg"] = True
        ffmpeg_version = result.stdout.split('\n')[0] if result.stdout else "unknown"
        logger.info(f"✅ FFmpeg обнаружен: {ffmpeg_version}")
    else:
        logger.warning("⚠️  FFmpeg не работает корректно. Проверьте установку.")
except (subprocess.SubprocessError, FileNotFoundError):
    logger.warning("⚠️  FFmpeg не установлен или не в PATH. Некоторые функции могут не работать.")

# Импорт нод
try:
    # Динамический импорт классов из модуля
    from . import audio_duration_node
    
    # Получаем NODE_CLASS_MAPPINGS и NODE_DISPLAY_NAME_MAPPINGS напрямую из модуля
    if hasattr(audio_duration_node, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS = getattr(audio_duration_node, 'NODE_CLASS_MAPPINGS', {})
    else:
        # Если нет глобальных переменных, создаем из классов
        NODE_CLASS_MAPPINGS = {}
        for attr_name in dir(audio_duration_node):
            attr = getattr(audio_duration_node, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'INPUT_TYPES'):
                # Это класс ноды ComfyUI
                NODE_CLASS_MAPPINGS[attr_name] = attr
    
    if hasattr(audio_duration_node, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS = getattr(audio_duration_node, 'NODE_DISPLAY_NAME_MAPPINGS', {})
    else:
        # Создаем отображаемые имена из имен классов
        NODE_DISPLAY_NAME_MAPPINGS = {}
        for class_name in NODE_CLASS_MAPPINGS.keys():
            # Преобразуем CamelCase в читаемый формат
            display_name = class_name
            if class_name.startswith('DVA_'):
                display_name = class_name[4:]  # Убираем префикс
            # Заменяем подчеркивания на пробелы
            display_name = display_name.replace('_', ' ')
            NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
    
    # Проверяем, что ноды загружены
    if not NODE_CLASS_MAPPINGS:
        logger.error("❌ Не удалось загрузить ноды. Проверьте файл audio_duration_node.py")
        raise ImportError("Failed to load node mappings")
    
    logger.info(f"✅ Успешно загружено {len(NODE_CLASS_MAPPINGS)} нод")
    
except ImportError as e:
    logger.error(f"❌ Ошибка импорта нод: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Создаем заглушки, чтобы ComfyUI не падал
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
except Exception as e:
    logger.error(f"❌ Неожиданная ошибка при загрузке нод: {str(e)}")
    logger.error(traceback.format_exc())
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Экспорт
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Вывод информации о загрузке
def print_welcome_message():
    """Вывод приветственного сообщения"""
    border = "=" * 60
    print(f"\n{border}")
    print("🎵 DVA AUDIO NODES - УСПЕШНО ЗАГРУЖЕНЫ 🎵".center(60))
    print(border)
    
    # Информация о нодах
    print(f"\n📊 ЗАГРУЖЕНО НОД: {len(NODE_CLASS_MAPPINGS)}")
    for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"   • {display_name}")
    
    # Информация о зависимостях
    print(f"\n🔧 ЗАВИСИМОСТИ:")
    for dep_name, available in DEPENDENCIES.items():
        status = "✅ ДОСТУПНО" if available else "❌ ОТСУТСТВУЕТ"
        print(f"   • {dep_name.upper():<10} : {status}")
    
    # Рекомендации
    missing_deps = [dep for dep, available in DEPENDENCIES.items() if not available and dep != "ffmpeg"]
    if missing_deps:
        print(f"\n⚠️  РЕКОМЕНДАЦИИ:")
        for dep in missing_deps:
            if dep == "pydub":
                print("   • Установите PyDub: pip install pydub")
            elif dep == "librosa":
                print("   • Установите Librosa: pip install librosa")
            elif dep == "soundfile":
                print("   • Установите SoundFile: pip install soundfile")
            elif dep == "numpy":
                print("   • Установите NumPy: pip install numpy")
    
    if not DEPENDENCIES["ffmpeg"]:
        print(f"\n⚠️  FFMPEG НЕ НАЙДЕН:")
        print("   • Ubuntu/Debian: sudo apt install ffmpeg")
        print("   • Windows: Скачайте с ffmpeg.org и добавьте в PATH")
        print("   • MacOS: brew install ffmpeg")
    
    print(f"\n📁 КАТЕГОРИИ В COMFYUI:")
    categories = set()
    for node_class in NODE_CLASS_MAPPINGS.values():
        if hasattr(node_class, 'CATEGORY'):
            categories.add(node_class.CATEGORY)
    
    for category in sorted(categories):
        print(f"   • {category}")
    
    print(f"\n{border}")
    print("🎧 Готово к работе! Перезапустите ComfyUI если не видите ноды.".center(60))
    print(border + "\n")

# Выводим информацию при загрузке
if COMFYUI_AVAILABLE and NODE_CLASS_MAPPINGS:
    print_welcome_message()
else:
    logger.error("❌ Не удалось инициализировать аудио ноды")