FROM python:3.9-slim

# Установка необходимых системных пакетов
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создаем директории для данных и логов
RUN mkdir -p data logs flask_session

# Установка правильных прав доступа
RUN chmod -R 755 /app && \
    chmod +x run_tests.py && \
    chown -R nobody:nogroup /app/data /app/logs /app/flask_session

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PORT=8000

# Проверка работоспособности
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Команда запуска
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} \
    --workers=1 \
    --threads=2 \
    --timeout=120 \
    --access-logfile=/app/logs/access.log \
    --error-logfile=/app/logs/error.log \
    --log-level=debug \
    --capture-output \
    --enable-stdio-inheritance \
    wsgi:app
