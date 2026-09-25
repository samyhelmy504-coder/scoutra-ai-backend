import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai


# ==========================================
# تحميل متغيرات البيئة
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY غير موجود في متغيرات البيئة أو ملف .env"
    )


# ==========================================
# الاتصال بـ Gemini
# ==========================================

client = genai.Client(
    api_key=api_key,
)


# ==========================================
# إنشاء SCOUTRA AI Backend
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
# Models
# ==========================================

class ChatRequest(BaseModel):
    message: str


# ==========================================
# الصفحة الرئيسية
# ==========================================

@app.get("/")
def home():
    return {
        "app": "SCOUTRA AI Backend",
        "status": "running",
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SCOUTRA AI Backend",
    }


# ==========================================
# SCOUTRA AI Chat
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    prompt = f"""
أنت SCOUTRA AI.

أنت مساعد ذكي متخصص في مجال الكشافة،
ومهمتك مساعدة الكشافين والقادة في:

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

    # ======================================
    # استدعاء Gemini مع تسجيل الخطأ الحقيقي
    # ======================================

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        return {
            "success": True,
            "reply": response.text,
        }

    except Exception as e:

        # إظهار الخطأ الحقيقي في Vercel Logs
        print("========================================")
        print("GEMINI ERROR:")
        print(repr(e))
        print("========================================")

        raise