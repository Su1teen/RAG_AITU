# University Chat Assistant API

## Установить зависимости:
    pip install -r requirements.txt

## Структура проекта
    ├── data/                  # Документы для учителей 
    ├── data_stud/             # Документы для студентов
    ├── indexes/               # Векторный индекс учителей
    ├── indexes_stud/          # Векторный индекс студентов
    ├── document_manager.py
    ├── document_processor.py
    ├── vector_storage.py
    ├── main.py                # приложение
    ├── .env
    └── README.md

## Запуск приложения
    uvicorn main:app --reload --host 0.0.0.0 --port 8000


## Основные API-эндпойнты
### Чат
POST: _/api/teacher/chat_
- Входной JSON:
    ```bash
    {
    "query": "Вопрос",
    "session_id": "default" 
    }

- Выходной JSON:
    ```bash
    {
    "answer": "Ответ от модели...\n\nSources:\n- файл.pdf (Page 1, 2)",
    "sources": ["- файл.pdf (Page 1, 2)"]
    }

__POST__: _/api/student/chat_: Формат аналогичный, используется набор данных и индекс для студентов

### Работа с документами
1. GET /api/teacher/docs
- Список активных документов для учителей.

2. GET /api/student/docs
- Список активных документов для студентов.

3. POST /api/teacher/docs/upload
- Загрузка документов (multipart/form-data).
#### Параметры:

- __file__ (файл)

- __title__ (опционально)

- __description__ (опционально)

- __tags__ (разделённая запятыми, опционально)

4. POST /api/student/docs/upload
- Аналогичная загрузка для студентов.

### Доп-ые эндпойнты

1. POST /api/{role}/chat/clear
- Очистка истории чата для заданной сессии.
    ```
    POST /api/teacher/chat/clear?session_id=default
2. GET /api/endpoints
- Возвращает список всех маршрутов приложения.