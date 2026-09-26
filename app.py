import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google import genai
from google.genai import types


# ==========================================
# Environment
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY غير موجود في متغيرات البيئة"
    )


# ==========================================
# Gemini Client
# ==========================================

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=9000,
        retry_options=types.HttpRetryOptions(
            attempts=1,
        ),
    ),
)


# ==========================================
# FastAPI
# ==========================================

app = FastAPI(
    title="SCOUTRA AI Backend",
    description="Backend الذكاء الاصطناعي لتطبيق SCOUTRA",
    version="1.0.0",
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Request
# ==========================================

class ChatRequest(BaseModel):
    message: str


# ==========================================
# Home
# ==========================================

@app.get("/")
def home():
    return {
        "app": "SCOUTRA AI Backend",
        "status": "running",
    }


# ==========================================
# Health
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SCOUTRA AI Backend",
    }


# ==========================================
# Gemini
# ==========================================

def ask_gemini(model: str, prompt: str):

    print(f"Trying {model}...")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    if not response.text:
        raise Exception(
            f"{model}: empty response"
        )

    print(f"{model} succeeded.")

    return response.text


# ==========================================
# Chat
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    prompt = f"""
أنت SCOUTRA AI.

أنت مساعد ذكي متخصص في مجال الكشافة.

مهمتك مساعدة الكشافين والقادة في:

- المهارات الكشفية
- الرحلات والخلوات
- التخييم
- الإسعافات الأولية بشكل تعليمي وآمن
- الملاحة والخرائط
- العقد والحبال
- الأنشطة والألعاب الكشفية
- المعلومات العامة المتعلقة بالكشافة
- تنظيم الأنشطة والفرق
- الإجابة على أسئلة الكشافين

أجب باللغة العربية بطريقة واضحة وبسيطة ومناسبة للكشاف.

سؤال المستخدم:
{request.message}
"""

    models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]

    errors = []

    for model in models:

        try:

            reply = ask_gemini(
                model=model,
                prompt=prompt,
            )

            return {
                "success": True,
                "model": model,
                "reply": reply,
            }

        except Exception as e:

            error_text = str(e)

            print("========================================")
            print(f"{model} FAILED")
            print(error_text)
            print("========================================")

            # نخزن معلومات الخطأ بدون أي API Key
            errors.append({
                "model": model,
                "error": error_text[:1000],
            })

            continue

    # ======================================
    # Diagnostic response
    # ======================================

    print("========================================")
    print("ALL GEMINI MODELS FAILED")
    print(errors)
    print("========================================")

    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "كل نماذج Gemini فشلت.",
            "models": errors,
        },
    )