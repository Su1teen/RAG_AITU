from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import openai  # или другой LLM-клиент, если используется
from dotenv import load_dotenv
from docx import Document
from docxtpl import DocxTemplate

router = APIRouter()
# Use absolute path to the correct tmp/generated directory
GENERATED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tmp', 'generated'))
os.makedirs(GENERATED_DIR, exist_ok=True)

TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates', 'practice_plan_template.docx'))

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/generate")
async def generate_file(request: Request):
    data = await request.json()
    description = data.get("description", "")
    # Генерация текста через LLM (OpenAI >= 1.0.0)
    try:
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Сгенерируй подробный отчет по следующему описанию задачи. Ответ дай в виде связного текста на русском языке."},
                {"role": "user", "content": description}
            ]
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"Ошибка генерации через LLM: {e}\n\nОписание: {description}"
    filename = f"{uuid.uuid4()}.docx"
    filepath = os.path.abspath(os.path.join(GENERATED_DIR, filename))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = Document()
    doc.add_paragraph(content)
    doc.save(filepath)
    print(f"[DEBUG] Saved DOCX to: {filepath}")
    return {"download_url": f"/api/download/{filename}"}

@router.post("/generate_practice_plan")
async def generate_practice_plan(request: Request):
    data = await request.json()
    # Ожидается: data = {"student_name": ..., "practice_type": ..., "start_date": ..., "end_date": ..., "tasks": ...}
    doc = DocxTemplate(TEMPLATE_PATH)
    doc.render(data)
    filename = f"practice_plan_{uuid.uuid4()}.docx"
    filepath = os.path.join(GENERATED_DIR, filename)
    doc.save(filepath)
    print(f"[DEBUG] Practice plan saved to: {filepath}")
    return {"download_url": f"/api/download/{filename}"}

@router.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.abspath(os.path.join(GENERATED_DIR, filename))
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found for download: {filepath}")
        return JSONResponse({"error": "File not found"}, status_code=404)
    print(f"[DEBUG] Downloading file: {filepath}")
    return FileResponse(filepath, filename=filename, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
