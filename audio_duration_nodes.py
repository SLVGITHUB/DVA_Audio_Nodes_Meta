"""
Audio Analysis Nodes for ComfyUI
Ноды для анализа и обработки аудио файлов
"""

import os
import json
import numpy as np
import torch
import folder_paths
import comfy.utils
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess
import tempfile
from datetime import datetime, timedelta

# Настройка логирования
logger = logging.getLogger(__name__)

# Проверка и импорт зависимостей
try:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Добавляем путь для аудио файлов
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.webm']

# Регистрируем типы файлов для ComfyUI
if hasattr(folder_paths, 'add_model_folder_path'):
    folder_paths.add_model_folder_path("audio_input", folder_paths.get_input_directory())

# ============================================================================
# 🎵 АУДИО - АНАЛИЗ ДЛИТЕЛЬНОСТИ
# ============================================================================

class DVA_Audio_Duration_Calculator:
    """🎵 Аудио - Анализ длительности"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Определение входных типов"""
        return {
            "required": {
                "audio": ("AUDIO",),
                "calculation_mode": (["accurate", "fast", "auto"], {"default": "auto"}),
                "time_precision": ("INT", {"default": 3, "min": 0, "max": 6, "step": 1}),
            },
            "optional": {
                "include_silence": ("BOOLEAN", {"default": True}),
                "silence_threshold_db": ("FLOAT", {"default": -60.0, "min": -100.0, "max": 0.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "JSON")
    RETURN_NAMES = ("duration_seconds", "duration_formatted", "status", "metadata")
    FUNCTION = "calculate_audio_duration"
    CATEGORY = "🎵 Audio/Analysis"
    DESCRIPTION = "Расчет длительности аудио файла"
    
    def calculate_audio_duration(self, audio, calculation_mode="auto", time_precision=3,
                                include_silence=True, silence_threshold_db=-60.0):
        """Основная функция расчета длительности"""
        
        try:
            # Логируем тип входных данных для отладки
            logger.info(f"Тип входных данных audio: {type(audio)}")
            
            # Проверяем формат AUDIO из ComfyUI (словарь с waveform и sample_rate)
            if isinstance(audio, dict):
                logger.info(f"Ключи словаря audio: {list(audio.keys())}")
                
                # Стандартный формат ComfyUI для аудио
                if 'waveform' in audio:
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 24000)
                    
                    logger.info(f"Получены аудио данные из словаря: форма waveform={waveform.shape if hasattr(waveform, 'shape') else 'unknown'}, sample_rate={sample_rate}")
                    
                    # Расчет длительности из тензора
                    duration = self._calculate_duration_from_tensor(waveform, sample_rate)
                    logger.info(f"Рассчитанная длительность из тензора: {duration} секунд")
                    
                    # Форматирование результатов
                    rounded_duration = round(duration, time_precision)
                    formatted_duration = self._format_duration(rounded_duration)
                    
                    # Подготовка метаданных
                    metadata = {
                        "calculation_method": "tensor_direct",
                        "sample_rate": sample_rate,
                        "waveform_shape": list(waveform.shape) if hasattr(waveform, 'shape') else [],
                        "time_precision": time_precision,
                        "include_silence": include_silence,
                        "silence_threshold_db": silence_threshold_db,
                        "total_samples": self._get_total_samples(waveform)
                    }
                    
                    # Если запрошено исключение тишины (заглушка)
                    if not include_silence and duration > 0:
                        # Здесь можно добавить логику удаления тишины
                        # Для простоты пока оставляем как есть
                        pass
                    
                    return (
                        float(rounded_duration),
                        formatted_duration,
                        "success",
                        json.dumps(metadata, ensure_ascii=False, indent=2)
                    )
                
                # Если это словарь с путем к файлу
                elif 'file_path' in audio:
                    audio_path = audio['file_path']
                    logger.info(f"Получен путь к файлу из словаря: {audio_path}")
                    return self._calculate_from_file(audio_path, calculation_mode, time_precision,
                                                    include_silence, silence_threshold_db)
            
            # Если это тензор напрямую
            elif torch.is_tensor(audio):
                logger.info(f"Получен тензор напрямую: форма={audio.shape}")
                # Используем стандартную частоту дискретизации по умолчанию
                duration = self._calculate_duration_from_tensor(audio, 24000)
                
                rounded_duration = round(duration, time_precision)
                formatted_duration = self._format_duration(rounded_duration)
                
                metadata = {
                    "calculation_method": "tensor_direct",
                    "sample_rate": 24000,
                    "waveform_shape": list(audio.shape),
                    "time_precision": time_precision,
                    "total_samples": self._get_total_samples(audio)
                }
                
                return (
                    float(rounded_duration),
                    formatted_duration,
                    "success",
                    json.dumps(metadata, ensure_ascii=False, indent=2)
                )
            
            # Если это список
            elif isinstance(audio, list) and len(audio) > 0:
                logger.info(f"Получен список, обрабатываем первый элемент")
                # Рекурсивно обрабатываем первый элемент
                return self.calculate_audio_duration(audio[0], calculation_mode, time_precision,
                                                     include_silence, silence_threshold_db)
            
            # Если это строка (путь к файлу)
            elif isinstance(audio, str):
                logger.info(f"Получен путь к файлу как строка: {audio}")
                return self._calculate_from_file(audio, calculation_mode, time_precision,
                                                include_silence, silence_threshold_db)
            
            # Если ничего не подошло
            logger.error(f"Неподдерживаемый тип аудио данных: {type(audio)}")
            return self._error_response(f"Неподдерживаемый тип аудио данных: {type(audio)}")
            
        except Exception as e:
            logger.error(f"Ошибка расчета длительности: {str(e)}", exc_info=True)
            return self._error_response(f"Исключение: {str(e)}")
    
    def _get_total_samples(self, tensor):
        """Получение общего количества семплов из тензора"""
        try:
            if tensor.dim() == 1:
                return tensor.shape[0]
            elif tensor.dim() == 2:
                return tensor.shape[1]
            elif tensor.dim() == 3:
                return tensor.shape[2]
            elif tensor.dim() == 4:
                return tensor.shape[3]
            else:
                return tensor.shape[-1]
        except:
            return 0
    
    def _calculate_duration_from_tensor(self, waveform, sample_rate):
        """Расчет длительности из тензора"""
        try:
            # Получаем количество семплов в зависимости от размерности тензора
            if waveform.dim() == 1:
                # Моно: [samples]
                num_samples = waveform.shape[0]
                logger.debug(f"Тензор 1D: {num_samples} семплов")
                
            elif waveform.dim() == 2:
                # [channels, samples] или [batch, samples]
                if waveform.shape[0] <= 2:  # Скорее всего [channels, samples]
                    num_samples = waveform.shape[1]
                    logger.debug(f"Тензор 2D [channels, samples]: {waveform.shape[0]} каналов, {num_samples} семплов")
                else:  # Скорее всего [batch, samples]
                    num_samples = waveform.shape[1]
                    logger.debug(f"Тензор 2D [batch, samples]: batch={waveform.shape[0]}, {num_samples} семплов")
                    
            elif waveform.dim() == 3:
                # [batch, channels, samples] или [1, channels, samples]
                num_samples = waveform.shape[2]
                logger.debug(f"Тензор 3D: batch={waveform.shape[0]}, channels={waveform.shape[1]}, {num_samples} семплов")
                
            elif waveform.dim() == 4:
                # [batch, channels, samples, 1] или другой формат
                num_samples = waveform.shape[2]
                logger.debug(f"Тензор 4D: {waveform.shape}")
                
            else:
                # Пробуем получить последнее измерение
                num_samples = waveform.shape[-1]
                logger.debug(f"Тензор {waveform.dim()}D, используем последнее измерение: {num_samples}")
            
            # Длительность = количество семплов / частота дискретизации
            if num_samples > 0 and sample_rate > 0:
                duration = num_samples / sample_rate
                logger.debug(f"Расчет: {num_samples} / {sample_rate} = {duration} сек")
                return duration
            else:
                logger.warning(f"Некорректные данные: num_samples={num_samples}, sample_rate={sample_rate}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка расчета из тензора: {str(e)}")
            # Пробуем альтернативный метод - через общее количество элементов
            try:
                total_elements = waveform.numel()
                # Предполагаем, что последнее измерение - это семплы
                # а остальные - batch и channels
                if waveform.dim() >= 1:
                    samples_per_channel = waveform.shape[-1]
                    num_channels = total_elements // samples_per_channel
                    logger.debug(f"Альтернативный расчет: total_elements={total_elements}, samples_per_channel={samples_per_channel}, num_channels={num_channels}")
                    
                    duration = samples_per_channel / sample_rate
                    return duration
            except:
                pass
            raise
    
    def _calculate_from_file(self, audio_path, calculation_mode, time_precision,
                            include_silence, silence_threshold_db):
        """Расчет длительности из файла"""
        
        if not audio_path:
            return self._error_response("Путь к аудио файлу не получен")
        
        if not os.path.exists(audio_path):
            return self._error_response(f"Аудио файл не найден по пути: {audio_path}")
        
        if not os.path.isfile(audio_path):
            return self._error_response(f"Путь не является файлом: {audio_path}")
        
        # Проверяем размер файла
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return self._error_response("Аудио файл пустой")
        
        logger.info(f"Размер файла: {file_size} байт, расширение: {os.path.splitext(audio_path)[1]}")
        
        # Выбор метода расчета
        if calculation_mode == "auto":
            calc_mode = self._select_calculation_mode(audio_path)
        else:
            calc_mode = calculation_mode
        
        logger.info(f"Выбран режим расчета: {calc_mode}")
        
        # Расчет длительности
        if calc_mode == "fast" and PYDUB_AVAILABLE:
            logger.info("Используем pydub для расчета")
            result = self._calculate_with_pydub(audio_path)
        elif LIBROSA_AVAILABLE:
            logger.info("Используем librosa для расчета")
            result = self._calculate_with_librosa(audio_path)
        else:
            logger.info("Используем ffmpeg для расчета")
            result = self._calculate_with_ffmpeg(audio_path)
        
        if result["status"] != "success":
            logger.error(f"Ошибка расчета: {result.get('error', 'Неизвестная ошибка')}")
            return self._error_response(result.get("error", "Ошибка расчета"))
        
        # Обработка тишины
        duration = result["duration"]
        logger.info(f"Исходная длительность из файла: {duration} секунд")
        
        if not include_silence and duration > 0:
            duration = self._remove_silence_duration(audio_path, duration, silence_threshold_db)
        
        # Форматирование результатов
        rounded_duration = round(duration, time_precision)
        formatted_duration = self._format_duration(rounded_duration)
        logger.info(f"Форматированная длительность: {formatted_duration}")
        
        # Подготовка метаданных
        metadata = result.get("metadata", {})
        metadata.update({
            "calculation_mode": calc_mode,
            "time_precision": time_precision,
            "include_silence": include_silence,
            "silence_threshold_db": silence_threshold_db,
            "file_path": audio_path,
            "file_size_bytes": file_size
        })
        
        return (
            float(rounded_duration),  # FLOAT
            formatted_duration,       # STRING
            "success",                # STRING
            json.dumps(metadata, ensure_ascii=False, indent=2)  # JSON
        )
    
    def _get_audio_path(self, audio_input):
        """Получение пути к аудио файлу из разных источников ComfyUI"""
        # Этот метод оставлен для обратной совместимости
        # Но теперь основная логика работает напрямую с тензорами
        
        logger.info(f"Тип входных данных в _get_audio_path: {type(audio_input)}")
        
        # 1. Если это строка (прямой путь)
        if isinstance(audio_input, str):
            logger.info(f"Входные данные - строка: {audio_input}")
            
            # Проверяем, существует ли файл
            if os.path.exists(audio_input):
                return audio_input
            
            # Если это имя файла без пути, ищем в input директории
            input_dir = folder_paths.get_input_directory()
            possible_path = os.path.join(input_dir, audio_input)
            if os.path.exists(possible_path):
                return possible_path
            
            # Ищем в output директории
            output_dir = folder_paths.get_output_directory()
            possible_path = os.path.join(output_dir, audio_input)
            if os.path.exists(possible_path):
                return possible_path
            
            # Ищем в temp директории
            temp_dir = folder_paths.get_temp_directory()
            possible_path = os.path.join(temp_dir, audio_input)
            if os.path.exists(possible_path):
                return possible_path
            
            return None
        
        # 2. Если это словарь (стандартный формат ComfyUI)
        elif isinstance(audio_input, dict):
            logger.info(f"Входные данные - словарь. Ключи: {list(audio_input.keys())}")
            
            # Пробуем разные возможные ключи
            possible_keys = ['file_path', 'filename', 'path', 'audio_path', 'input_path']
            
            for key in possible_keys:
                if key in audio_input and audio_input[key]:
                    path = audio_input[key]
                    logger.info(f"Найден ключ '{key}': {path}")
                    
                    if os.path.exists(path):
                        return path
                    
                    # Если путь относительный, ищем в стандартных директориях
                    input_dir = folder_paths.get_input_directory()
                    possible_path = os.path.join(input_dir, path)
                    if os.path.exists(possible_path):
                        return possible_path
            
            return None
        
        return None
    
    def _select_calculation_mode(self, audio_path):
        """Автоматический выбор метода расчета"""
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext in ['.mp3', '.aac', '.m4a'] and PYDUB_AVAILABLE:
            return "fast"
        elif file_ext in ['.wav', '.flac'] and LIBROSA_AVAILABLE:
            return "accurate"
        else:
            return "accurate"
    
    def _calculate_with_pydub(self, audio_path):
        """Расчет с использованием pydub"""
        try:
            logger.info(f"Загружаем аудио через pydub: {audio_path}")
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # в секундах
            
            info = mediainfo(audio_path)
            metadata = {
                "method": "pydub",
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "bit_depth": audio.sample_width * 8,
                "bitrate": int(info.get('bit_rate', 0)) if info else 0,
                "format": os.path.splitext(audio_path)[1].lower(),
                "file_size_bytes": os.path.getsize(audio_path),
                "codec": info.get('codec_name', 'unknown')
            }
            
            logger.info(f"Pydub рассчитал длительность: {duration} секунд")
            return {"status": "success", "duration": duration, "metadata": metadata}
        except Exception as e:
            error_msg = f"Pydub error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _calculate_with_librosa(self, audio_path):
        """Расчет с использованием librosa"""
        try:
            logger.info(f"Загружаем аудио через librosa: {audio_path}")
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)
            
            metadata = {
                "method": "librosa",
                "sample_rate": sr,
                "channels": y.shape[0] if len(y.shape) > 1 else 1,
                "duration_samples": y.shape[-1],
                "format": os.path.splitext(audio_path)[1].lower(),
                "file_size_bytes": os.path.getsize(audio_path)
            }
            
            logger.info(f"Librosa рассчитала длительность: {duration} секунд")
            return {"status": "success", "duration": duration, "metadata": metadata}
        except Exception as e:
            error_msg = f"Librosa error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _calculate_with_ffmpeg(self, audio_path):
        """Расчет с использованием ffmpeg"""
        try:
            logger.info(f"Загружаем аудио через ffmpeg: {audio_path}")
            
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                   'format=duration:stream=duration,sample_rate,channels,codec_name',
                   '-of', 'json', audio_path]
            
            logger.info(f"Выполняем команду: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_msg = f"FFprobe error: {result.stderr}"
                logger.error(error_msg)
                return {"status": "error", "error": error_msg}
            
            data = json.loads(result.stdout)
            logger.debug(f"FFprobe данные: {json.dumps(data, indent=2)}")
            
            # Получаем длительность из формата или потоков
            duration = 0.0
            if 'format' in data and 'duration' in data['format']:
                duration = float(data['format']['duration'])
            elif 'streams' in data and data['streams']:
                for stream in data['streams']:
                    if 'duration' in stream:
                        stream_duration = float(stream['duration'])
                        duration = max(duration, stream_duration)
            
            if duration == 0.0:
                return {"status": "error", "error": "Не удалось определить длительность"}
            
            metadata = {
                "method": "ffmpeg",
                "format": os.path.splitext(audio_path)[1].lower(),
                "file_size_bytes": os.path.getsize(audio_path),
                "ffprobe_data": data
            }
            
            logger.info(f"FFmpeg рассчитал длительность: {duration} секунд")
            return {"status": "success", "duration": duration, "metadata": metadata}
            
        except subprocess.TimeoutExpired:
            error_msg = "FFprobe timeout"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        except json.JSONDecodeError as e:
            error_msg = f"FFprobe JSON decode error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
        except Exception as e:
            error_msg = f"FFprobe exception: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    def _remove_silence_duration(self, audio_path, total_duration, threshold_db):
        """Расчет длительности без тишины"""
        # Заглушка - можно реализовать детектирование тишины
        # через pydub.detect_silence или librosa.effects.split
        return total_duration
    
    def _format_duration(self, seconds):
        """Форматирование длительности в ЧЧ:ММ:СС.ммм"""
        if seconds <= 0:
            return "00:00:00.000"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{minutes:02d}:{secs:06.3f}"
    
    def _error_response(self, error_message):
        """Формирование ответа об ошибке"""
        logger.error(f"Возвращаем ошибку: {error_message}")
        return (0.0, "00:00:00.000", f"error: {error_message}", "{}")


# ============================================================================
# 🎵 АУДИО - ЗАГРУЗКА ФАЙЛА (ДОПОЛНИТЕЛЬНАЯ НОДА)
# ============================================================================

class DVA_Load_Audio_File:
    """🎵 Аудио - Загрузка файла"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """Определение входных типов"""
        input_dir = folder_paths.get_input_directory()
        audio_files = []
        
        # Ищем аудио файлы во всех поддиректориях
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    audio_files.append(rel_path)
        
        return {
            "required": {
                "audio_file": (sorted(audio_files), {"audio_upload": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "load_audio"
    CATEGORY = "🎵 Audio/Input"
    DESCRIPTION = "Загрузка аудио файла"
    
    def load_audio(self, audio_file):
        """Загрузка аудио файла"""
        try:
            input_dir = folder_paths.get_input_directory()
            full_path = os.path.join(input_dir, audio_file)
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Файл не найден: {full_path}")
            
            # Возвращаем путь к файлу в формате AUDIO
            return (
                {"file_path": full_path, "filename": audio_file},
                full_path
            )
            
        except Exception as e:
            logger.error(f"Ошибка загрузки аудио: {str(e)}")
            return ({"file_path": "", "filename": ""}, "")


# ============================================================================
# 🎵 АУДИО - МЕТАДАННЫЕ
# ============================================================================

class DVA_Audio_Metadata_Extractor:
    """🎵 Аудио - Извлечение метаданных"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "extract_format": ("BOOLEAN", {"default": True}),
                "extract_technical": ("BOOLEAN", {"default": True}),
                "extract_tags": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("JSON", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("metadata", "summary", "format", "duration")
    FUNCTION = "extract_audio_metadata"
    CATEGORY = "🎵 Audio/Analysis"
    DESCRIPTION = "Извлечение метаданных из аудио файла"
    
    def extract_audio_metadata(self, audio, extract_format=True, extract_technical=True, extract_tags=False):
        """Извлечение метаданных"""
        try:
            # Проверяем, есть ли данные в формате AUDIO
            if isinstance(audio, dict) and 'waveform' in audio:
                # Это тензорные данные, извлекаем базовую информацию
                waveform = audio['waveform']
                sample_rate = audio.get('sample_rate', 24000)
                
                # Получаем длительность из тензора
                duration = self._get_duration_from_tensor(waveform, sample_rate)
                
                metadata = {
                    "format": {
                        "format_name": "tensor",
                        "sample_rate": sample_rate
                    },
                    "technical": {
                        "duration_seconds": duration,
                        "sample_rate": sample_rate,
                        "channels": waveform.shape[1] if waveform.dim() >= 2 and waveform.shape[1] <= 2 else 1,
                        "total_samples": waveform.shape[-1] if waveform.dim() > 0 else 0
                    },
                    "file_info": {
                        "name": "audio_tensor",
                        "size_bytes": waveform.element_size() * waveform.nelement()
                    }
                }
                
                # Создание сводки
                summary = self._create_metadata_summary(metadata)
                
                return (
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    summary,
                    "tensor",
                    float(duration)
                )
            
            # Иначе пытаемся получить путь к файлу
            calculator = DVA_Audio_Duration_Calculator()
            audio_path = calculator._get_audio_path(audio)
            
            if not audio_path or not os.path.exists(audio_path):
                return ("{}", "Error: Audio file not found", "unknown", 0.0)
            
            metadata = self._extract_all_metadata(audio_path, extract_format, extract_technical, extract_tags)
            
            # Создание сводки
            summary = self._create_metadata_summary(metadata)
            
            # Основные поля
            audio_format = metadata.get("format", {}).get("format_name", "unknown")
            duration = metadata.get("technical", {}).get("duration_seconds", 0.0)
            
            return (
                json.dumps(metadata, ensure_ascii=False, indent=2),  # JSON
                summary,                                              # STRING
                audio_format,                                         # STRING
                float(duration)                                       # FLOAT
            )
            
        except Exception as e:
            logger.error(f"Ошибка извлечения метаданных: {e}")
            return ("{}", f"Error: {str(e)}", "error", 0.0)
    
    def _get_duration_from_tensor(self, waveform, sample_rate):
        """Получение длительности из тензора"""
        try:
            if waveform.dim() == 1:
                num_samples = waveform.shape[0]
            elif waveform.dim() == 2:
                num_samples = waveform.shape[1]
            elif waveform.dim() == 3:
                num_samples = waveform.shape[2]
            else:
                num_samples = waveform.shape[-1]
            
            return num_samples / sample_rate
        except:
            return 0.0
    
    def _extract_all_metadata(self, audio_path, extract_format, extract_technical, extract_tags):
        """Извлечение всех метаданных"""
        metadata = {
            "file_info": {
                "path": audio_path,
                "name": os.path.basename(audio_path),
                "size_bytes": os.path.getsize(audio_path),
                "size_mb": round(os.path.getsize(audio_path) / (1024*1024), 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(audio_path)).isoformat()
            }
        }
        
        if extract_format or extract_technical:
            ffprobe_data = self._get_ffprobe_metadata(audio_path)
            
            if extract_format:
                metadata["format"] = ffprobe_data.get("format", {})
            
            if extract_technical:
                metadata["technical"] = self._extract_technical_metadata(ffprobe_data)
        
        if extract_tags and PYDUB_AVAILABLE:
            try:
                from pydub.utils import mediainfo
                info = mediainfo(audio_path)
                metadata["tags"] = {k: v for k, v in info.items() 
                                  if k not in ['format_name', 'duration', 'bit_rate', 
                                               'sample_rate', 'channels']}
            except:
                metadata["tags"] = {}
        
        return metadata
    
    def _get_ffprobe_metadata(self, audio_path):
        """Получение метаданных через ffprobe"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                   '-show_format', '-show_streams', audio_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except:
            pass
        
        return {"format": {}, "streams": []}
    
    def _extract_technical_metadata(self, ffprobe_data):
        """Извлечение технических метаданных"""
        tech_data = {}
        
        if 'format' in ffprobe_data:
            fmt = ffprobe_data['format']
            tech_data.update({
                "duration_seconds": float(fmt.get('duration', 0)),
                "bitrate_bps": int(fmt.get('bit_rate', 0)),
                "size_bytes": int(fmt.get('size', 0))
            })
        
        # Ищем аудио потоки
        audio_streams = [s for s in ffprobe_data.get('streams', []) 
                        if s.get('codec_type') == 'audio']
        
        if audio_streams:
            stream = audio_streams[0]
            tech_data.update({
                "sample_rate": int(stream.get('sample_rate', 0)),
                "channels": int(stream.get('channels', 1)),
                "codec": stream.get('codec_name', 'unknown'),
                "bits_per_sample": stream.get('bits_per_sample', 0)
            })
        
        return tech_data
    
    def _create_metadata_summary(self, metadata):
        """Создание текстовой сводки"""
        parts = []
        
        file_info = metadata.get("file_info", {})
        if file_info:
            parts.append(f"{file_info.get('name', 'Unknown')} "
                        f"({file_info.get('size_mb', 0)} MB)")
        
        tech = metadata.get("technical", {})
        if tech:
            duration = tech.get("duration_seconds", 0)
            if duration > 0:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                secs = duration % 60
                
                if hours > 0:
                    dur_str = f"{hours}:{minutes:02d}:{secs:04.1f}"
                else:
                    dur_str = f"{minutes}:{secs:04.1f}"
                
                parts.append(f"Duration: {dur_str}")
            
            if tech.get("sample_rate"):
                parts.append(f"{tech['sample_rate']} Hz")
            
            if tech.get("channels"):
                ch = tech['channels']
                parts.append(f"{ch}ch")
        
        return " | ".join(parts)


# ============================================================================
# 🎵 АУДИО - ПАКЕТНАЯ ОБРАБОТКА
# ============================================================================

class DVA_Audio_Batch_Processor:
    """🎵 Аудио - Пакетная обработка"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False}),
                "file_pattern": ("STRING", {"default": "*.mp3,*.wav,*.flac", "multiline": False}),
                "operation": (["duration", "metadata", "both"], {"default": "both"}),
                "recursive": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("JSON", "STRING", "LIST")
    RETURN_NAMES = ("results", "summary", "file_list")
    FUNCTION = "process_audio_batch"
    CATEGORY = "🎵 Audio/Batch"
    DESCRIPTION = "Пакетная обработка аудио файлов в директории"
    
    def process_audio_batch(self, directory_path, file_pattern="*.mp3,*.wav,*.flac",
                           operation="both", recursive=True):
        """Пакетная обработка файлов"""
        try:
            if not directory_path or not os.path.isdir(directory_path):
                return ("{}", "Error: Directory not found", "[]")
            
            # Поиск файлов
            audio_files = self._find_audio_files(directory_path, file_pattern, recursive)
            
            if not audio_files:
                return ("{}", "No audio files found", "[]")
            
            # Обработка файлов
            results = []
            duration_calc = DVA_Audio_Duration_Calculator()
            
            for audio_file in audio_files:
                try:
                    file_result = self._process_single_file(audio_file, operation, duration_calc)
                    results.append(file_result)
                except Exception as e:
                    results.append({
                        "file": audio_file,
                        "error": str(e),
                        "status": "failed"
                    })
            
            # Создание сводки
            summary = self._create_batch_summary(results, operation)
            
            return (
                json.dumps(results, ensure_ascii=False, indent=2),  # JSON
                summary,                                            # STRING
                json.dumps(audio_files, ensure_ascii=False)         # LIST (as JSON string)
            )
            
        except Exception as e:
            logger.error(f"Ошибка пакетной обработки: {e}")
            return ("{}", f"Error: {str(e)}", "[]")
    
    def _find_audio_files(self, directory, pattern, recursive):
        """Поиск аудио файлов"""
        import fnmatch
        import glob
        
        patterns = [p.strip() for p in pattern.split(',')]
        audio_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file_pattern in patterns:
                    audio_files.extend(
                        os.path.join(root, f) for f in fnmatch.filter(files, file_pattern)
                    )
        else:
            for file_pattern in patterns:
                audio_files.extend(glob.glob(os.path.join(directory, file_pattern)))
        
        return sorted(list(set(audio_files)))
    
    def _process_single_file(self, audio_file, operation, duration_calculator):
        """Обработка одного файла"""
        result = {
            "file": audio_file,
            "filename": os.path.basename(audio_file),
            "size_bytes": os.path.getsize(audio_file),
            "status": "success"
        }
        
        # Расчет длительности
        if operation in ["duration", "both"]:
            try:
                duration_result = duration_calculator.calculate_audio_duration(
                    {"file_path": audio_file},
                    calculation_mode="auto"
                )
                result["duration_seconds"] = duration_result[0]
                result["duration_formatted"] = duration_result[1]
            except Exception as e:
                result["duration_error"] = str(e)
        
        # Извлечение метаданных
        if operation in ["metadata", "both"]:
            try:
                metadata_extractor = DVA_Audio_Metadata_Extractor()
                meta_result = metadata_extractor.extract_audio_metadata(
                    {"file_path": audio_file}
                )
                result["metadata"] = json.loads(meta_result[0])
            except Exception as e:
                result["metadata_error"] = str(e)
        
        return result
    
    def _create_batch_summary(self, results, operation):
        """Создание сводки по пакетной обработке"""
        total = len(results)
        successful = len([r for r in results if r.get("status") == "success"])
        
        if operation in ["duration", "both"]:
            total_duration = sum(r.get("duration_seconds", 0) for r in results 
                               if "duration_seconds" in r)
            
            if total_duration < 60:
                dur_str = f"{total_duration:.1f}s"
            elif total_duration < 3600:
                dur_str = f"{total_duration/60:.1f}m"
            else:
                dur_str = f"{total_duration/3600:.1f}h"
                
            return (f"Processed {successful}/{total} files | "
                   f"Total duration: {dur_str}")
        else:
            return f"Processed {successful}/{total} files"


# ============================================================================
# ЭКСПОРТ КЛАССОВ
# ============================================================================

NODE_CLASS_MAPPINGS = {
    # Загрузка аудио
    "DVA_Load_Audio_File": DVA_Load_Audio_File,
    
    # Основные ноды анализа
    "DVA_Audio_Duration_Calculator": DVA_Audio_Duration_Calculator,
    "DVA_Audio_Metadata_Extractor": DVA_Audio_Metadata_Extractor,
    
    # Пакетная обработка
    "DVA_Audio_Batch_Processor": DVA_Audio_Batch_Processor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Загрузка аудио
    "DVA_Load_Audio_File": "DVA 🎵 Аудио - Загрузка файла",
    
    # Основные ноды анализа
    "DVA_Audio_Duration_Calculator": "DVA 🎵 Аудио - Анализ длительности",
    "DVA_Audio_Metadata_Extractor": "DVA 🎵 Аудио - Извлечение метаданных",
    
    # Пакетная обработка
    "DVA_Audio_Batch_Processor": "DVA 🎵 Аудио - Пакетная обработка",
}
