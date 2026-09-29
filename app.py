import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
        timeout=10000,
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
# Request Models
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str

    # سجل المحادثة السابقة
    history: list[ChatMessage] = Field(
        default_factory=list
    )


# ==========================================
# SCOUTRA Knowledge Base
# ==========================================

KNOWLEDGE_FILE = (
    Path(__file__).resolve().parent
    / "scoutra_jawalah_knowledge.json"
)


def load_knowledge_base():
    try:
        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        chunks = data.get("chunks", [])

        print(
            "========================================"
        )
        print(
            "SCOUTRA KNOWLEDGE BASE LOADED"
        )
        print(
            f"Chunks: {len(chunks)}"
        )
        print(
            "========================================"
        )

        return chunks

    except Exception as e:

        print(
            "========================================"
        )
        print(
            "WARNING: KNOWLEDGE BASE FAILED"
        )
        print(str(e))
        print(
            "========================================"
        )

        return []


KNOWLEDGE_BASE = load_knowledge_base()


# ==========================================
# Arabic Normalization
# ==========================================

def normalize_arabic(text: str) -> str:

    text = text.lower()

    # إزالة التشكيل
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u065F\u0670]",
        "",
        text,
    )

    # توحيد أشكال الحروف
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # إزالة علامات الترقيم
    text = re.sub(
        r"[^\w\s\u0600-\u06FF]",
        " ",
        text,
    )

    # توحيد المسافات
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ==========================================
# Extract Words
# ==========================================

def extract_words(text: str) -> set[str]:

    normalized = normalize_arabic(text)

    words = normalized.split()

    return {
        word
        for word in words
        if len(word) >= 3
    }


# ==========================================
# Search Knowledge Base
# ==========================================

def search_knowledge(
    query: str,
    max_results: int = 5,
):

    if not KNOWLEDGE_BASE:
        return []

    query_normalized = normalize_arabic(
        query
    )

    query_words = extract_words(
        query
    )

    if not query_words:
        return []

    scored_chunks = []

    for chunk in KNOWLEDGE_BASE:

        text = str(
            chunk.get("text", "")
        )

        page = chunk.get("page")

        chunk_id = chunk.get(
            "id",
            "",
        )

        if not text.strip():
            continue

        normalized_text = normalize_arabic(
            text
        )

        text_words = extract_words(
            text
        )

        if not text_words:
            continue

        # الكلمات المشتركة
        common_words = (
            query_words.intersection(
                text_words
            )
        )

        score = len(
            common_words
        )

        # لو السؤال نفسه أو جزء كبير منه
        # موجود داخل النص
        if (
            len(query_normalized) >= 8
            and query_normalized
            in normalized_text
        ):
            score += 5

        # مصطلحات كشفية مهمة
        important_terms = [
            "جواله",
            "كشافة",
            "كشفيه",
            "كشفية",
            "رحله",
            "تخييم",
            "رهط",
            "عشير",
            "شاره",
            "نشاط",
            "منهج",
            "قائد",
            "بادن",
            "باول",
            "وعد",
            "قانون",
            "اسعاف",
            "مغامره",
        ]

        for term in important_terms:

            if term in query_normalized:

                if term in normalized_text:
                    score += 1

        if score > 0:

            scored_chunks.append(
                {
                    "score": score,
                    "page": page,
                    "id": chunk_id,
                    "text": text,
                }
            )

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # لازم يكون فيه تطابق معقول
    results = [
        item
        for item in scored_chunks
        if item["score"] >= 2
    ]

    return results[:max_results]


# ==========================================
# Build Knowledge Context
# ==========================================

def build_knowledge_context(
    question: str,
):

    results = search_knowledge(
        question,
        max_results=5,
    )

    if not results:
        return "", []

    parts = []

    parts.append(
        """
مراجع من المنهج الكشفي الخاص بـ SCOUTRA:

استخدم المعلومات التالية كمصدر مرجعي
للأسئلة المرتبطة بالمنهج الكشفي.

قواعد استخدام المرجع:

- إذا كان السؤال مرتبطًا بالمنهج
  والمراجع الموجودة تجيب عنه،
  اعتمد عليها.

- لا تنسب للمنهج معلومة غير موجودة فيه.

- إذا كانت معلومات المنهج غير كافية،
  يمكنك استخدام معرفتك العامة لإكمال الإجابة.

- إذا كان السؤال خارج موضوع المنهج،
  استخدم معرفتك العامة بشكل طبيعي.

- لا تجعل عدم وجود الإجابة في المنهج
  يمنعك من الإجابة.

- إذا ذكرت معلومة محددة من المنهج،
  يمكنك ذكر رقم الصفحة عندما يكون ذلك مفيدًا.
"""
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        parts.append(
            f"""
--- مرجع {index}
الصفحة: {result["page"]}

{result["text"]}
"""
        )

    return (
        "\n".join(parts),
        results,
    )


# ==========================================
# Home
# ==========================================

@app.get("/")
def home():

    return {
        "app": "SCOUTRA AI Backend",
        "status": "running",
        "knowledge_base": len(
            KNOWLEDGE_BASE
        ),
    }


# ==========================================
# Health
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "SCOUTRA AI Backend",
        "knowledge_base": len(
            KNOWLEDGE_BASE
        ),
    }


# ==========================================
# SCOUTRA AI System Instructions
# ==========================================

SYSTEM_PROMPT = """
أنت SCOUTRA AI.

هويتك:

أنت مساعد ذكاء اصطناعي متخصص في مجال الكشافة،
تم تطويرك بواسطة جرجس سامي من كشافة رئيس
الملائكة ميخائيل بأسيوط.

إذا سألك المستخدم عن:

- من أنشأك؟
- من طورك؟
- مين صاحب SCOUTRA AI؟
- مين عملك؟
- مين مطورك؟

أجب بوضوح:

"تم تطوير SCOUTRA AI بواسطة جرجس سامي
من كشافة رئيس الملائكة ميخائيل بأسيوط."

لا تذكر معلومات المطور من نفسك
إلا إذا كان السؤال متعلقًا بهويتك
أو مطورك.


أسلوبك:

- تحدث بالعربية الواضحة والبسيطة.
- كن ودودًا ومحترمًا.
- اجعل إجاباتك مناسبة للكشافين والقادة.
- لا تستخدم إجابات طويلة بدون داعٍ.
- إذا كان السؤال يحتاج شرحًا،
  رتبه في نقاط واضحة.
- لا تدّعي أنك إنسان.
- لا تخترع معلومات غير مؤكدة.
- إذا لم تعرف الإجابة،
  قل بوضوح إنك غير متأكد.


تخصصك:

- المهارات الكشفية
- الرحلات والخلوات
- التخييم
- الإسعافات الأولية بشكل تعليمي وآمن
- الملاحة والخرائط
- العقد والحبال
- الأنشطة والألعاب الكشفية
- المعلومات العامة المتعلقة بالكشافة
- تنظيم الأنشطة والفرق
- التقاليد والمبادئ الكشفية


المنهج الكشفي:

لديك قاعدة معرفة تحتوي على المنهج
الكشفي المستخدم في SCOUTRA.

عندما يتم تزويدك بمراجع من المنهج:

- اعتبر المنهج مصدرًا أساسيًا للأسئلة
  المرتبطة به.

- حافظ على مصطلحات المنهج وطريقته
  قدر الإمكان.

- لا تنسب للمنهج معلومة غير موجودة فيه.

- إذا لم يقدم المنهج إجابة كافية،
  استخدم معرفتك العامة وأكمل الإجابة.

- إذا كان السؤال خارج المنهج،
  استخدم الذكاء الاصطناعي العام بشكل طبيعي.

- لا تقل للمستخدم إنك تستخدم قاعدة بيانات
  داخلية إلا إذا كان ذلك مفيدًا.


سياق المحادثة:

لديك سياق للمحادثة السابقة يتم إرساله
مع كل رسالة.

استخدم هذا السياق لفهم:

- كلام المستخدم السابق.
- الضمائر.
- الإشارات مثل "ده" و"دي".
- "الموضوع اللي كنا بنتكلم عنه".
- الأسئلة التي تعتمد على إجابات سابقة.

إذا كان المستخدم يتابع موضوعًا سابقًا،
استمر في نفس الموضوع بدل أن تطلب منه
إعادة شرحه، ما دام السياق السابق واضحًا.


السلامة:

في الإسعافات الأولية والسلامة:

قدم معلومات تعليمية عامة وآمنة.

عند وجود حالة خطيرة أو طارئة،
شجع المستخدم على طلب المساعدة
من شخص بالغ أو مختص أو خدمات الطوارئ.
"""


# ==========================================
# Convert Chat History
# ==========================================

def build_contents(
    history: list[ChatMessage],
    current_message: str,
):
    """
    Build a Gemini request that is always guaranteed to end with a user turn.

    Gemini can reject a request when the supplied conversation history has
    an invalid turn sequence (for example, consecutive model turns or a
    history that effectively ends on a model turn). To keep SCOUTRA AI's
    memory reliable, we normalize the previous conversation into a single
    user-context block and then send the current message as the final user
    turn.
    """

    context_lines = []

    for item in history:
        role = item.role.lower().strip()
        content = item.content.strip()

        if not content:
            continue

        if role == "assistant" or role == "model":
            speaker = "SCOUTRA AI"
        elif role == "user":
            speaker = "المستخدم"
        else:
            continue

        context_lines.append(f"{speaker}: {content}")

    contents = []

    if context_lines:
        history_text = "\n".join(context_lines)

        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "هذا سجل المحادثة السابقة للمحافظة على السياق. "
                            "اعتبره سياقًا سابقًا وليس سؤالًا جديدًا.\n\n"
                            + history_text
                        )
                    )
                ],
            )
        )

    # الرسالة الحالية يجب أن تكون آخر turn، وبصفة user.
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=current_message.strip()
                )
            ],
        )
    )

    return contents


# ==========================================
# Gemini
# ==========================================

def ask_gemini(
    model: str,
    contents,
    system_instruction: str,
):

    print(
        f"Trying {model}..."
    )

    response = (
        client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
    )

    if not response.text:

        raise Exception(
            f"{model}: empty response"
        )

    print(
        f"{model} succeeded."
    )

    return response.text


# ==========================================
# Chat
# ==========================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    print(
        "========================================"
    )

    print(
        "NEW CHAT REQUEST"
    )

    print(
        f"Current message: "
        f"{request.message}"
    )

    print(
        f"History messages: "
        f"{len(request.history)}"
    )

    print(
        "========================================"
    )


    # ======================================
    # Search Knowledge Base
    # ======================================

    (
        knowledge_context,
        knowledge_results,
    ) = build_knowledge_context(
        request.message
    )


    if knowledge_results:

        print(
            "Knowledge base matches: "
            f"{len(knowledge_results)}"
        )

        for result in knowledge_results:

            print(
                f"- Page "
                f"{result['page']} "
                f"(score="
                f"{result['score']})"
            )

    else:

        print(
            "No knowledge base match."
        )

        print(
            "Using general AI knowledge."
        )


    # ======================================
    # Build System Instruction
    # ======================================

    system_instruction = (
        SYSTEM_PROMPT
    )

    if knowledge_context:

        system_instruction += (
            "\n\n"
            + knowledge_context
        )


    # ======================================
    # Build Conversation Context
    # ======================================

    contents = build_contents(
        history=request.history,
        current_message=request.message,
    )


    # ======================================
    # Fallback Models
    # ======================================

    models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]

    errors = []


    # ======================================
    # Try Models
    # ======================================

    for model in models:

        try:

            reply = ask_gemini(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
            )

            return {
                "success": True,
                "model": model,
                "reply": reply,
                "knowledge_used": bool(
                    knowledge_results
                ),
                "knowledge_pages": [
                    result["page"]
                    for result
                    in knowledge_results
                ],
            }

        except Exception as e:

            error_text = str(e)

            print(
                "========================================"
            )

            print(
                f"{model} FAILED"
            )

            print(
                error_text
            )

            print(
                "========================================"
            )

            errors.append(
                {
                    "model": model,
                    "error": error_text[:1000],
                }
            )

            # أخطاء INVALID_ARGUMENT عادةً تكون بسبب شكل الطلب نفسه،
            # وبالتالي إعادة المحاولة بنموذج آخر لن تصلح المشكلة.
            if "INVALID_ARGUMENT" in error_text or "Requests ending with a model turn" in error_text:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "طلب SCOUTRA AI غير صالح.",
                        "details": error_text[:1000],
                    },
                )

            continue


    # ======================================
    # All Models Failed
    # ======================================

    print(
        "========================================"
    )

    print(
        "ALL GEMINI MODELS FAILED"
    )

    print(
        errors
    )

    print(
        "========================================"
    )


    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": "كل نماذج Gemini فشلت.",
            "models": errors,
        },
    )