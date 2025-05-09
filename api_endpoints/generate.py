from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import openai  # или другой LLM-клиент, если используется
from dotenv import load_dotenv

router = APIRouter()
GENERATED_DIR = "tmp/generated"
os.makedirs(GENERATED_DIR, exist_ok=True)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/generate")
async def generate_file(request: Request):
    data = await request.json()
    description = data.get("description", "")
    # Генерация текста через LLM (пример для OpenAI, замените на ваш LLM)
    try:
        llm_response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Сгенерируй подробный отчет по следующему описанию задачи. Ответ дай в виде связного текста на русском языке."},
                {"role": "user", "content": description}
            ]
        )
        content = llm_response.choices[0].message.content.strip()
    except Exception as e:
        content = f"Ошибка генерации через LLM: {e}\n\nОписание: {description}"
    filename = f"{uuid.uuid4()}.txt"
    filepath = os.path.join(GENERATED_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return {"download_url": f"/api/download/{filename}"}

@router.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(filepath, filename=filename, media_type="text/plain; charset=utf-8")
