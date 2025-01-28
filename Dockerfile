FROM python:3.9-slim

# Установка необходимых системных пакетов
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Делаем скрипт для тестов исполняемым
RUN chmod +x run_tests.py

# Создаем директории для данных и логов
RUN mkdir -p data logs

# Установка переменных окружения по умолчанию
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Проверка работоспособности приложения
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Запуск тестов при сборке (опционально)
# RUN python run_tests.py

# Команда запуска
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} \
    --workers=4 \
    --threads=2 \
    --timeout=120 \
    --access-logfile=logs/access.log \
    --error-logfile=logs/error.log \
    --log-level=info \
    app:app
