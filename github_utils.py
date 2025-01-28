from github import Github, GithubException, Repository, InputGitTreeElement
import os
import base64
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import json
from pathlib import Path
import hashlib
import time
from threading import Lock

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('github_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubSync:
    """Класс для синхронизации данных с GitHub."""
    
    def __init__(self):
        self.token = os.environ.get("AI_PROGRESS_TOKEN")
        self.username = os.environ.get("GITHUB_USERNAME", "Azerus96")
        self.repository = os.environ.get("GITHUB_REPOSITORY", "finalofc")
        self.branch = "main"
        self._github: Optional[Github] = None
        self._repo: Optional[Repository] = None
        self._lock = Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_sync: Dict[str, float] = {}
        self.sync_interval = 300  # 5 минут между синхронизациями

    @property
    def github(self) -> Github:
        """Получает экземпляр Github с ленивой инициализацией."""
        if not self._github:
            if not self.token:
                raise ValueError("GitHub token not set in environment variables")
            self._github = Github(self.token)
        return self._github

    @property
    def repo(self) -> Repository:
        """Получает экземпляр репозитория с ленивой инициализацией."""
        if not self._repo:
            try:
                self._repo = self.github.get_user(self.username).get_repo(self.repository)
            except GithubException as e:
                logger.error(f"Error accessing repository: {e}")
                raise
        return self._repo

    def save_progress(self, filename: str, data: Any, commit_message: str = None) -> bool:
        """
        Сохраняет прогресс в GitHub репозиторий.
        
        Args:
            filename: Имя файла
            data: Данные для сохранения
            commit_message: Сообщение коммита

        Returns:
            bool: Успешность операции
        """
        if not self._should_sync(filename):
            logger.debug(f"Skipping sync for {filename} - too soon since last sync")
            return False

        with self._lock:
            try:
                # Подготавливаем данные
                if isinstance(data, (dict, list)):
                    content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
                else:
                    content = base64.b64encode(pickle.dumps(data))

                # Создаем хеш данных
                data_hash = hashlib.sha256(content).hexdigest()[:8]
                
                # Формируем сообщение коммита
                if not commit_message:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    commit_message = f"Update {filename} [hash: {data_hash}] at {timestamp}"

                try:
                    # Проверяем существование файла
                    file_exists = True
                    try:
                        contents = self.repo.get_contents(filename, ref=self.branch)
                        current_hash = hashlib.sha256(base64.b64decode(contents.content)).hexdigest()[:8]
                        
                        # Проверяем, изменились ли данные
                        if current_hash == data_hash:
                            logger.debug(f"File {filename} hasn't changed, skipping update")
                            return True
                            
                    except GithubException as e:
                        if e.status == 404:
                            file_exists = False
                        else:
                            raise

                    # Создаем или обновляем файл
                    if file_exists:
                        self.repo.update_file(
                            contents.path,
                            commit_message,
                            content,
                            contents.sha,
                            branch=self.branch
                        )
                    else:
                        self.repo.create_file(
                            filename,
                            commit_message,
                            content,
                            branch=self.branch
                        )

                    # Обновляем кэш и время синхронизации
                    self._cache[filename] = data
                    self._last_sync[filename] = time.time()
                    
                    logger.info(f"Successfully saved {filename} to GitHub")
                    return True

                except GithubException as e:
                    logger.error(f"GitHub API error while saving {filename}: {e}")
                    return False

            except Exception as e:
                logger.error(f"Unexpected error while saving {filename}: {e}")
                return False

    def load_progress(self, filename: str, use_cache: bool = True) -> Optional[Any]:
        """
        Загружает прогресс из GitHub репозитория.
        
        Args:
            filename: Имя файла
            use_cache: Использовать ли кэш

        Returns:
            Optional[Any]: Загруженные данные или None в случае ошибки
        """
        try:
            # Проверяем кэш
            if use_cache and filename in self._cache:
                logger.debug(f"Returning cached data for {filename}")
                return self._cache[filename]

            contents = self.repo.get_contents(filename, ref=self.branch)
            content = base64.b64decode(contents.content)

            # Пытаемся загрузить как JSON
            try:
                data = json.loads(content.decode('utf-8'))
            except json.JSONDecodeError:
                # Если не JSON, пытаемся загрузить как pickle
                try:
                    data = pickle.loads(content)
                except:
                    logger.error(f"Could not decode content of {filename}")
                    return None

            # Обновляем кэш
            self._cache[filename] = data
            self._last_sync[filename] = time.time()
            
            return data

        except GithubException as e:
            if e.status == 404:
                logger.warning(f"File {filename} not found in repository")
            else:
                logger.error(f"GitHub API error while loading {filename}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error while loading {filename}: {e}")
            return None

    def _should_sync(self, filename: str) -> bool:
        """Проверяет, нужно ли синхронизировать файл."""
        last_sync = self._last_sync.get(filename, 0)
        return time.time() - last_sync >= self.sync_interval

    def create_backup(self, filename: str) -> bool:
        """Создает резервную копию файла в репозитории."""
        try:
            contents = self.repo.get_contents(filename, ref=self.branch)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"backups/{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
            
            self.repo.create_file(
                backup_filename,
                f"Backup of {filename}",
                contents.content,
                branch=self.branch
            )
            
            logger.info(f"Created backup of {filename} as {backup_filename}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup of {filename}: {e}")
            return False

    def cleanup_old_backups(self, keep_last: int = 5) -> None:
        """Удаляет старые резервные копии, оставляя указанное количество последних."""
        try:
            contents = self.repo.get_contents("backups", ref=self.branch)
            backups = sorted(
                [c for c in contents if c.path.startswith("backups/")],
                key=lambda x: x.path
            )

            # Удаляем старые резервные копии
            for backup in backups[:-keep_last]:
                self.repo.delete_file(
                    backup.path,
                    f"Remove old backup {backup.path}",
                    backup.sha,
                    branch=self.branch
                )
                logger.info(f"Removed old backup {backup.path}")

        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

# Создаем глобальный экземпляр синхронизатора
github_sync = GitHubSync()

def save_progress_to_github(filename: str, data: Any = None, commit_message: str = None) -> bool:
    """Обертка для сохранения прогресса."""
    return github_sync.save_progress(filename, data, commit_message)

def load_progress_from_github(filename: str, use_cache: bool = True) -> Optional[Any]:
    """Обертка для загрузки прогресса."""
    return github_sync.load_progress(filename, use_cache)

def create_backup_on_github(filename: str) -> bool:
    """Обертка для создания резервной копии."""
    return github_sync.create_backup(filename)

def cleanup_github_backups(keep_last: int = 5) -> None:
    """Обертка для очистки старых резервных копий."""
    github_sync.cleanup_old_backups(keep_last)
