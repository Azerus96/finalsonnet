import pickle
import json
import os
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
import numpy as np
from pathlib import Path
import shutil
import hashlib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    """Класс для управления данными игры и статистикой."""
    
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.cache: Dict[str, Any] = {}
        self.backup_dir = self.base_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)

    def save_data(self, data: Any, filename: str, use_backup: bool = True) -> bool:
        """
        Сохраняет данные в файл с опциональным резервным копированием.
        
        Args:
            data: Данные для сохранения
            filename: Имя файла
            use_backup: Создавать ли резервную копию

        Returns:
            bool: Успешность операции
        """
        try:
            file_path = self.base_dir / filename
            
            # Создаем резервную копию если файл существует
            if use_backup and file_path.exists():
                self._create_backup(file_path)

            # Определяем формат файла и сохраняем соответственно
            if filename.endswith('.pkl'):
                self._save_pickle(data, file_path)
            elif filename.endswith('.json'):
                self._save_json(data, file_path)
            else:
                raise ValueError(f"Unsupported file format for {filename}")

            # Обновляем кэш
            self.cache[filename] = data
            
            logger.info(f"Successfully saved data to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving data to {filename}: {e}")
            return False

    def load_data(self, filename: str, use_cache: bool = True) -> Optional[Any]:
        """
        Загружает данные из файла с поддержкой кэширования.
        
        Args:
            filename: Имя файла
            use_cache: Использовать ли кэш

        Returns:
            Optional[Any]: Загруженные данные или None в случае ошибки
        """
        try:
            # Проверяем кэш
            if use_cache and filename in self.cache:
                logger.debug(f"Returning cached data for {filename}")
                return self.cache[filename]

            file_path = self.base_dir / filename
            if not file_path.exists():
                logger.warning(f"File {filename} not found")
                return None

            # Загружаем данные в зависимости от формата
            if filename.endswith('.pkl'):
                data = self._load_pickle(file_path)
            elif filename.endswith('.json'):
                data = self._load_json(file_path)
            else:
                raise ValueError(f"Unsupported file format for {filename}")

            # Обновляем кэш
            if use_cache:
                self.cache[filename] = data

            return data

        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            return None

    def _create_backup(self, file_path: Path) -> None:
        """Создает резервную копию файла."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        shutil.copy2(file_path, backup_path)
        
        # Удаляем старые резервные копии (оставляем последние 5)
        self._cleanup_old_backups(file_path.stem)

    def _cleanup_old_backups(self, file_stem: str) -> None:
        """Удаляет старые резервные копии, оставляя только последние 5."""
        backups = sorted(
            self.backup_dir.glob(f"{file_stem}_*"),
            key=lambda x: x.stat().st_mtime
        )
        for backup in backups[:-5]:  # Оставляем последние 5 копий
            backup.unlink()

    def _save_pickle(self, data: Any, file_path: Path) -> None:
        """Сохраняет данные в формате pickle."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def _save_json(self, data: Any, file_path: Path) -> None:
        """Сохраняет данные в формате JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_pickle(self, file_path: Path) -> Any:
        """Загружает данные из формата pickle."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _load_json(self, file_path: Path) -> Any:
        """Загружает данные из формата JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

class StatisticsManager:
    """Класс для управления статистикой игры."""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.stats_file = 'game_statistics.json'
        self.current_stats = self.load_statistics()

    def load_statistics(self) -> Dict:
        """Загружает статистику из файла."""
        stats = self.data_manager.load_data(self.stats_file)
        return stats if stats else {
            'games_played': 0,
            'fantasies': 0,
            'fouls': 0,
            'royalties': 0,
            'win_rates': {},
            'hand_statistics': {},
            'player_progress': []
        }

    def update_statistics(self, game_result: Dict) -> None:
        """Обновляет статистику на основе результатов игры."""
        try:
            self.current_stats['games_played'] += 1
            
            # Обновляем базовую статистику
            self.current_stats['fantasies'] += game_result.get('fantasies', 0)
            self.current_stats['fouls'] += game_result.get('fouls', 0)
            self.current_stats['royalties'] += game_result.get('royalties', 0)

            # Обновляем статистику рук
            hand_type = self._categorize_hand(game_result)
            self.current_stats['hand_statistics'][hand_type] = \
                self.current_stats['hand_statistics'].get(hand_type, 0) + 1

            # Добавляем прогресс игрока
            self.current_stats['player_progress'].append({
                'timestamp': datetime.now().isoformat(),
                'score': game_result.get('score', 0),
                'hand_type': hand_type
            })

            # Сохраняем обновленную статистику
            self.save_statistics()
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def _categorize_hand(self, game_result: Dict) -> str:
        """Категоризирует руку на основе результатов игры."""
        if game_result.get('fantasy'):
            return 'fantasy'
        if game_result.get('foul'):
            return 'foul'
        if game_result.get('royalties', 0) > 10:
            return 'high_royalty'
        if game_result.get('royalties', 0) > 0:
            return 'low_royalty'
        return 'standard'

    def save_statistics(self) -> None:
        """Сохраняет текущую статистику в файл."""
        self.data_manager.save_data(self.current_stats, self.stats_file)

    def get_summary(self) -> Dict:
        """Возвращает сводку статистики."""
        return {
            'total_games': self.current_stats['games_played'],
            'fantasy_rate': self.current_stats['fantasies'] / max(1, self.current_stats['games_played']),
            'foul_rate': self.current_stats['fouls'] / max(1, self.current_stats['games_played']),
            'avg_royalties': self.current_stats['royalties'] / max(1, self.current_stats['games_played']),
            'hand_distribution': self.current_stats['hand_statistics']
        }

# Создаем глобальные экземпляры менеджеров
data_manager = DataManager()
statistics_manager = StatisticsManager(data_manager)

def save_data(data: Any, filename: str) -> bool:
    """Обертка для сохранения данных."""
    return data_manager.save_data(data, filename)

def load_data(filename: str) -> Optional[Any]:
    """Обертка для загрузки данных."""
    return data_manager.load_data(filename)

def update_statistics(game_result: Dict) -> None:
    """Обертка для обновления статистики."""
    statistics_manager.update_statistics(game_result)

def get_statistics_summary() -> Dict:
    """Обертка для получения сводки статистики."""
    return statistics_manager.get_summary()
