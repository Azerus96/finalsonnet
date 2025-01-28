#!/usr/bin/env python3
import pytest
import os
import sys

def main():
    """Запускает тесты с настройками для Render.com."""
    # Настраиваем переменные окружения для тестов
    os.environ['TESTING'] = 'True'
    os.environ['AI_PROGRESS_TOKEN'] = os.environ.get('AI_PROGRESS_TOKEN', 'dummy_token')
    os.environ['GITHUB_USERNAME'] = os.environ.get('GITHUB_USERNAME', 'test_user')
    os.environ['GITHUB_REPOSITORY'] = os.environ.get('GITHUB_REPOSITORY', 'test_repo')

    # Добавляем текущую директорию в PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # Запускаем тесты
    args = [
        '--verbose',
        '--cov=.',
        '--cov-report=html',
        '--cov-report=term',
        'tests'
    ]
    
    return pytest.main(args)

if __name__ == '__main__':
    sys.exit(main())
