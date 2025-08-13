#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import pickle
import hashlib
import asyncio
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

from fastapi import FastAPI, Request, Response
from telegram import Update, Document, ReplyKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document as DocxDocument

from google import genai
from google.genai import types

import requests
import io

# ------------ ЛОГИ ------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-bot")
for noisy in ("httpx", "google_genai", "google_genai.models"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ------------ КОНФИГ ------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан")

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY не задан")

# Пути
DOC_FOLDER = os.getenv("DOC_FOLDER", "/tmp/documents")
INDEX_FILE = os.getenv("FAISS_INDEX", "/tmp/index.faiss")
TEXTS_FILE = os.getenv("TEXTS_FILE", "/tmp/texts.pkl")
MANIFEST_FILE = os.getenv("MANIFEST_FILE", "/tmp/manifest.json")
USAGE_FILE = os.getenv("USAGE_FILE", "/tmp/usage.json")

DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_LIMIT", "10"))
ADMIN_USER_IDS = set(int(x.strip()) for x in os.getenv("ADMIN_USER_IDS", "").split(",") if x.strip().isdigit())

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

# --- параметры умного деления на чанки ---
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "400"))
SUBCHUNK_MAX_CHARS = int(os.getenv("SUBCHUNK_MAX_CHARS", "1600"))
TELEGRAM_MSG_LIMIT = 4096

# ------------ ТОВАРНЫЕ ЗНАКИ (Google Sheets) ------------
TM_SHEET_ID = os.getenv("TM_SHEET_ID", "").strip()
TM_SHEET_NAME = os.getenv("TM_SHEET_NAME", "Лист1").strip()
TM_SHEET_GID = os.getenv("TM_SHEET_GID", "0").strip()
TM_ENABLE = os.getenv("TM_ENABLE", "1") == "1"
TM_SHEET_CSV_URL = os.getenv("TM_SHEET_CSV_URL", "").strip()
TM_DEBUG = os.getenv("TM_DEBUG", "0") == "1"

# ------------ Режим запуска ------------
RUN_MODE = os.getenv("RUN_MODE", "polling").strip().lower()  # "polling" | "webhook"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# ------------ КНОПКИ UI ------------
AI_LABEL = "🤖 AI-чат"
DOCS_LABEL = "📄 Вопросы по документам"
TM_LABEL = "🏷️ Товарные знаки"
MAIN_KB = ReplyKeyboardMarkup([[AI_LABEL, DOCS_LABEL, TM_LABEL]], resize_keyboard=True)

TM_LABELS = ['№', 'Номер заявки', 'Номер регистрации', '', 'Описание', 'Статус', 'Срок действия', 'Комментарии', '', 'Ссылка']

# Гарантируем каталоги
def _ensure_dir(path: str) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        return False

if not _ensure_dir(DOC_FOLDER):
    base = "/opt/render/project/src/data"
    if not _ensure_dir(base):
        base = "/tmp/data"
        _ensure_dir(base)
    logger.warning("Нет доступа к %s. Переключаемся на %s", DOC_FOLDER, base)
    DOC_FOLDER   = os.path.join(base, "documents")
    INDEX_FILE   = os.path.join(base, "index.faiss")
    TEXTS_FILE   = os.path.join(base, "texts.pkl")
    MANIFEST_FILE= os.path.join(base, "manifest.json")
    _ensure_dir(DOC_FOLDER)

Path(Path(INDEX_FILE).parent).mkdir(parents=True, exist_ok=True)
Path(Path(TEXTS_FILE).parent).mkdir(parents=True, exist_ok=True)
Path(Path(MANIFEST_FILE).parent).mkdir(parents=True, exist_ok=True)
Path(Path(USAGE_FILE).parent).mkdir(parents=True, exist_ok=True)

# ------------ КЛИЕНТ GEMINI ------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------ ХЕЛПЕРЫ ЛИМИТА ------------
def _today_str() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())

def _load_usage() -> dict:
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Не удалось прочитать USAGE_FILE: %s", e)
    return {}

def _save_usage(d: dict):
    try:
        with open(USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Не удалось записать USAGE_FILE: %s", e)

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_USER_IDS

def get_usage(user_id: int) -> int:
    data = _load_usage()
    today = _today_str()
    return int(data.get(str(user_id), {}).get(today, 0))

def inc_usage(user_id: int) -> int:
    data = _load_usage()
    today = _today_str()
    u = data.setdefault(str(user_id), {})
    u[today] = int(u.get(today, 0)) + 1
    _save_usage(data)
    return u[today]

# ------------ ФАЙЛЫ/МАНИФЕСТ ------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(m: dict):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def load_texts() -> List[str]:
    if os.path.exists(TEXTS_FILE):
        with open(TEXTS_FILE, "rb") as f:
            return pickle.load(f)
    return []

def save_texts(texts: List[str]):
    with open(TEXTS_FILE, "wb") as f:
        pickle.dump(texts, f)

def load_index() -> faiss.Index | None:
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return None

def save_index(index: faiss.Index):
    faiss.write_index(index, INDEX_FILE)

# ------------ ИЗВЛЕЧЕНИЕ ТЕКСТА ------------
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        pages_text = []
        for page in doc:
            txt = page.get_text().strip()
            pages_text.append(txt)
        text = "\n".join(pages_text).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PyMuPDF не смог извлечь текст (%s). Переходим к OCR.", repr(e))
    try:
        images = convert_from_path(file_path)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n".join(ocr_texts).strip()
    except Exception as e:
        logger.error("Ошибка OCR/convert_from_path: %s", repr(e))
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = DocxDocument(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        logger.error("Ошибка чтения DOCX: %s", repr(e))
        return ""

def extract_text_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp1251", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        logger.error("Ошибка чтения TXT: %s", repr(e))
        return ""

# ------------ УМНОЕ ДЕЛЕНИЕ НА ЧАНКИ ------------
CONTRACT_SECTIONS = [
    "Предмет договора","Права и обязанности сторон","Обязанности сторон","Гарантии сторон","Ответственность сторон",
    "Срок действия договора","Финансовые условия","Стоимость услуг и порядок оплаты","Порядок оплаты","Термины и определения",
    "Прочие условия","Обстоятельства непреодолимой силы","Форс-мажор","Конфиденциальность","Право на использование изображений",
    "Порядок использования результатов интеллектуальной деятельности","Заверения обстоятельства",
    "Адрес и банковские реквизиты","Реквизиты и подписи сторон","Подписи сторон"
]
POSITION_SECTIONS = [
    "Общие положения","Цели и задачи","Предмет регулирования","Термины и определения","Функции","Права и обязанности",
    "Права организации","Обязанности организации","Ответственность","Порядок взаимодействия","Организация работы",
    "Порядок внесения изменений","Заключительные положения"
]
GENERIC_SECTIONS = [
    "Введение","Общие положения","Термины и определения","Порядок","Права и обязанности","Права","Обязанности","Ответственность",
    "Срок действия","Финансовые условия","Порядок оплаты","Конфиденциальность","Прочие условия","Заключительные положения","Приложения"
]

HEAD_NUM_RE = re.compile(r"^(?:раздел|глава|section|chapter)\s+\d+[.:)]?$", re.IGNORECASE|re.MULTILINE)
HEAD_NUM2_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S+", re.MULTILINE)
HEAD_ROMAN_RE = re.compile(r"^(?:[IVXLCDM]+)[\.\)]\s+\S+", re.IGNORECASE|re.MULTILINE)

def guess_doc_type(text: str) -> str:
    head = text[:5000].lower()
    if "договор" in head:
        return "contract"
    if "положение" in head:
        return "position"
    return "generic"

def is_all_caps_cyr(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    upp = sum(1 for ch in letters if ch.upper() == ch and "а" <= ch.lower() <= "я")
    return upp >= max(3, int(0.7 * len(letters)))

def find_headings(text: str, headings: List[str]) -> List[tuple[int,str]]:
    res = []
    for h in headings:
        for m in re.finditer(rf"(?im)^\s*{re.escape(h)}\s*$", text):
            res.append((m.start(), h))
    for m in HEAD_NUM_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()]))
    for m in HEAD_NUM2_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()].strip()))
    for m in HEAD_ROMAN_RE.finditer(text):
        res.append((m.start(), text[m.start():m.end()].strip()))
    for m in re.finditer(r"(?m)^(?P<line>.+)$", text):
        line = m.group("line")
        if is_all_caps_cyr(line):
            res.append((m.start(), line.strip()))
    uniq = {}
    for off, ttl in res:
        uniq[off] = ttl
    return sorted(uniq.items(), key=lambda x: x[0])

def split_by_sections(text: str, headings: List[str]) -> List[tuple[str,str]]:
    marks = find_headings(text, headings)
    if not marks:
        return [("", text.strip())]
    chunks = []
    for i, (start, title) in enumerate(marks):
        end = marks[i+1][0] if i+1 < len(marks) else len(text)
        chunk = text[start:end].strip()
        lines = chunk.splitlines()
        if lines:
            first_line = lines[0].strip()
            if len(first_line) <= 180:
                title = first_line
        chunks.append((title.strip(), chunk))
    return chunks

def split_long_chunk(title: str, body: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out = []
    cur = title + "\n"
    for p in paras:
        add_len = 2 + len(p) if cur.strip() != title else len(p)
        if len(cur) + add_len > SUBCHUNK_MAX_CHARS and cur.strip():
            out.append(cur.strip())
            cur = title + "\n" + p
        else:
            if cur.strip() == title:
                cur = title + "\n" + p
            else:
                cur += "\n\n" + p
    if cur.strip():
        out.append(cur.strip())
    return out

def smart_split_text(text: str) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []
    text = re.sub(r"\r\n?", "\n", text)

    doc_type = guess_doc_type(text)
    base_sections = CONTRACT_SECTIONS if doc_type=="contract" else POSITION_SECTIONS if doc_type=="position" else GENERIC_SECTIONS

    section_chunks = split_by_sections(text, base_sections)

    normalized: List[str] = []
    buf = ""
    for title, chunk in section_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) < CHUNK_MIN_CHARS and buf:
            buf += "\n\n" + chunk
            continue
        if buf:
            normalized.append(buf)
        buf = chunk
    if buf:
        normalized.append(buf)

    final_chunks: List[str] = []
    for ch in normalized:
        if len(ch) <= CHUNK_MAX_CHARS:
            final_chunks.append(ch)
        else:
            lines = ch.splitlines()
            title = lines[0].strip() if lines else "Раздел"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ch
            parts = split_long_chunk(title, body)
            for p in parts:
                if len(p) > CHUNK_MAX_CHARS:
                    sents = re.split(r"(?<=[.!?…])\s+", p)
                    tmp = ""
                    for s in sents:
                        if len(tmp) + len(s) + 1 > CHUNK_MAX_CHARS:
                            if tmp.strip():
                                final_chunks.append(tmp.strip())
                            tmp = s
                        else:
                            tmp = (tmp + " " + s).strip()
                    if tmp.strip():
                        final_chunks.append(tmp.strip())
                else:
                    final_chunks.append(p.strip())

    if not final_chunks:
        sents = re.split(r"(?<=[.!?…])\s+", text)
        tmp = ""
        for s in sents:
            if len(tmp) + len(s) + 1 > CHUNK_MAX_CHARS:
                if tmp.strip():
                    final_chunks.append(tmp.strip())
                tmp = s
            else:
                tmp = (tmp + " " + s).strip()
        if tmp.strip():
            final_chunks.append(tmp.strip())

    return [c for c in final_chunks if c and c.strip()]

# ------------ ЭМБЕДДИНГИ/ИНДЕКС ------------
def get_embedding(text: str) -> np.ndarray:
    max_attempts = 5
    base = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
            vec = resp.embeddings[0].values
            return np.array(vec, dtype="float32")
        except Exception as e:
            msg = repr(e)
            transient = any(code in msg for code in ["503", "UNAVAILABLE", "429", "502", "504"])
            if attempt < max_attempts and transient:
                sleep_s = base * (2 ** (attempt - 1)) + random.uniform(0, 0.15)
                logger.warning("Эмбеддинг временно недоступен (%s). Повтор #%d через %.1f c", msg, attempt, sleep_s)
                time.sleep(sleep_s)
                continue
            logger.exception("Ошибка эмбеддинга (после %d попыток): %s", attempt, msg)
            raise

def ensure_index(dim: int) -> faiss.Index:
    index = load_index()
    if index is None:
        index = faiss.IndexFlatL2(dim)
    return index

def index_file(file_path: str) -> Tuple[int, int]:
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "docx":
        text = extract_text_from_docx(file_path)
    elif ext == "txt":
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Неподдерживаемое расширение файла.")
    if not text:
        logger.warning("Пустой текст после извлечения: %s", os.path.basename(file_path))
        return (0, 0)

    chunks = smart_split_text(text)
    if not chunks:
        return (0, 0)

    texts = load_texts()
    first_vec = None
    for ch in chunks:
        try:
            first_vec = get_embedding(ch)
            break
        except Exception as e:
            logger.warning("Пропущен чанк при определении размерности: %s", repr(e))
            continue
    if first_vec is None:
        raise RuntimeError("Не удалось получить эмбеддинги.")
    index = ensure_index(len(first_vec))
    new_embeddings, new_texts = [], []
    for ch in chunks:
        try:
            emb = get_embedding(ch)
            new_embeddings.append(emb)
            new_texts.append(ch)
            time.sleep(0.02)
        except Exception as e:
            logger.warning("Пропущен чанк из-за ошибки эмбеддинга: %s", repr(e))
            continue
    if not new_embeddings:
        existing = load_index()
        return (0, existing.ntotal if existing else 0)
    mat = np.vstack(new_embeddings).astype("float32")
    index.add(mat)
    save_index(index)
    texts.extend(new_texts)
    save_texts(texts)
    total = index.ntotal if hasattr(index, "ntotal") else len(texts)
    return (len(new_texts), total)

def retrieve_chunks(query: str, k: int = RETRIEVAL_K) -> List[str]:
    if not (os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE)):
        return []
    q_emb = get_embedding(query)
    index = load_index()
    texts = load_texts()
    if index is None or len(texts) == 0:
        return []
    D, I = index.search(np.array([q_emb], dtype="float32"), k=min(k, len(texts)))
    ids = [i for i in I[0] if 0 <= i < len(texts)]
    return [texts[i] for i in ids]

# ------------ ГЕНЕРАЦИЯ ------------
def generate_answer_with_gemini(user_query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join(retrieved_chunks[:RETRIEVAL_K]) if retrieved_chunks else "(контекст не найден)"
    prompt = (
        "Вы — юридический помощник. Дай развёрнутый, практичный ответ.\n\n"
        "ИСПОЛЬЗУЙ ТОЛЬКО факты из Контекста ниже. Если чего-то нет в Контексте — прямо скажи, не выдумывай.\n\n"
        "Структура ответа:\n"
        "1) Краткий итог в 2–4 строках.\n"
        "2) Подробный разбор по пунктам, указывая ИМЯ документа, где содержатся положения.\n"
        "3) Цитаты из документа.\n"
        "4) Чёткие шаги.\n\n"
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ЗАПРОС:\n{user_query}"
    )
    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        if getattr(resp, "text", None):
            return resp.text.strip()
        pf = getattr(resp, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            return f"⚠️ Запрос отклонён модерацией: {pf.block_reason}."
        cands = getattr(resp, "candidates", []) or []
        for c in cands:
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                parts_text = "".join(getattr(p, "text", "") for p in c.content.parts)
                if parts_text.strip():
                    return parts_text.strip()
        return "⚠️ Ответ пуст."
    except Exception as e:
        msg = repr(e)
        if any(x in msg for x in ["429", "503", "502", "504", "UNAVAILABLE", "ResourceExhausted"]):
            return "⚠️ Перегрузка модели. Повторите позже."
        if "401" in msg or "403" in msg or "PermissionDenied" in msg:
            return "⚠️ Неверный ключ или нет доступа."
        return f"⚠️ Ошибка: {msg}"

def generate_direct_ai_answer(user_query: str) -> str:
    system = (
        "Ты — внимательный и полезный ассистент. Отвечай чётко, по делу. "
        "Если вопрос юридический и у пользователя нет документов, давай общий совет и предупреждай о необходимости проверки юристом."
    )
    prompt = f"СИСТЕМА:\n{system}\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
    try:
        resp = client.models.generate_content(
            model=TEXT_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        if getattr(resp, "text", None):
            return resp.text.strip()
        return "⚠️ Ответ пуст."
    except Exception as e:
        msg = repr(e)
        if any(x in msg for x in ["429", "503", "502", "504", "UNAVAILABLE", "ResourceExhausted"]):
            return "⚠️ Перегрузка модели. Повторите позже."
        if "401" in msg or "403" in msg or "PermissionDenied" in msg:
            return "⚠️ Неверный ключ или нет доступа."
        return f"⚠️ Ошибка: {msg}"

def _split_for_telegram(text: str, max_len: int = TELEGRAM_MSG_LIMIT - 200) -> list[str]:
    parts, buf = [], []
    cur = 0
    paragraphs = text.replace("\r\n", "\n").split("\n\n")
    for p in paragraphs:
        p = p.strip()
        if not p:
            candidate = "\n\n".join(buf).strip()
            if candidate:
                parts.append(candidate)
                buf, cur = [], 0
            continue
        delta = len(p) + (2 if cur > 0 else 0)
        if cur + delta <= max_len:
            buf.append(p); cur += delta
        else:
            candidate = "\n\n".join(buf).strip()
            if candidate:
                parts.append(candidate)
            buf, cur = [], 0
            while len(p) > max_len:
                parts.append(p[:max_len]); p = p[max_len:]
            if p:
                buf = [p]; cur = len(p)
    candidate = "\n\n".join(buf).strip()
    if candidate:
        parts.append(candidate)
    return parts

async def send_long(update: Update, text: str):
    for chunk in _split_for_telegram(text):
        await update.message.reply_text(chunk, disable_web_page_preview=True)

# ------------ TM: URL/IMAGE HELPERS ------------
def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#039;"))

URL_RE = re.compile(r'(https?://[^\s<>")]+)', re.IGNORECASE)

def _extract_urls(cell: str) -> list[str]:
    """Достаём ВСЕ URL из ячейки (поддержка нескольких ссылок)."""
    txt = str(cell or "")
    tokens = re.split(r'[,\s]+', txt.strip())
    urls = []
    for t in tokens:
        if t.startswith("http://") or t.startswith("https://"):
            urls.append(t)
        else:
            urls.extend(URL_RE.findall(t))
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _normalize_image_url(url: str) -> str:
    """Нормализация Google Drive ссылок -> прямые ссылки на файл."""
    u = url.strip()
    m = re.search(r'drive\.google\.com/file/d/([^/]+)/', u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r'(?:[?&]id=)([A-Za-z0-9_-]+)', u)
    if m and "drive.google.com" in u:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return u

def _is_probable_image_url(url: str) -> bool:
    u = url.lower()
    if any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return True
    if "googleusercontent.com" in u or "uc?export=download" in u or "=download" in u:
        return True
    return False

def _format_date(value: str) -> str:
    from datetime import datetime
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%d/%m/%Y")
        except Exception:
            pass
    return value.strip()

def tm_format_row(row: list[str], labels: list[str] = TM_LABELS) -> tuple[str, list[str]]:
    formatted_lines = []
    image_urls: list[str] = []

    for idx, val in enumerate(row):
        label = labels[idx] if idx < len(labels) else ""
        if idx == 3:
            continue

        cell = str(val or "").strip()
        if not cell:
            continue

        # 1) вытаскиваем все URL из ячейки
        urls = _extract_urls(cell)
        for u in urls:
            nu = _normalize_image_url(u)
            if _is_probable_image_url(nu):
                image_urls.append(nu)

        # 2) если ячейка — только ссылки, не дублируем их в тексте
        only_links = urls and (cell == " ".join(urls) or cell == ",".join(urls))
        if only_links:
            continue

        if re.match(r"^\d{1,4}[-./]\d{1,2}[-./]\d{1,4}$", cell):
            cell = _format_date(cell)

        if label:
            formatted_lines.append(f"<b>{_html_escape(label)}:</b> {_html_escape(cell)}")
        else:
            formatted_lines.append(_html_escape(cell))

    # дедуп картинок
    seen, uniq_images = set(), []
    for u in image_urls:
        if u not in seen:
            seen.add(u); uniq_images.append(u)

    text = "\n".join(formatted_lines).strip()
    return text, uniq_images

async def _tm_send_image_safely(chat_id: int, url: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Сначала пробуем URL напрямую; если не вышло — скачиваем и отправляем байты."""
    try:
        await context.bot.send_photo(chat_id, photo=url)
        return True
    except Exception as e:
        logger.warning("TM: send_photo by URL failed for %s: %r", url, e)

    try:
        r = requests.get(url, timeout=25)
        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code == 200 and (r.content and ("image/" in ct or len(r.content) > 0)):
            buf = io.BytesIO(r.content)
            filename = "image"
            ul = url.lower()
            if ".png" in ul: filename += ".png"
            elif ".jpg" in ul or ".jpeg" in ul: filename += ".jpg"
            elif ".webp" in ul: filename += ".webp"
            elif ".gif" in ul: filename += ".gif"
            else:
                if "image/png" in ct: filename += ".png"
                elif "image/jpeg" in ct: filename += ".jpg"
                elif "image/webp" in ct: filename += ".webp"
                elif "image/gif" in ct: filename += ".gif"
                else: filename += ".bin"
            await context.bot.send_photo(chat_id, photo=InputFile(buf, filename=filename))
            return True
    except Exception as e:
        logger.warning("TM: fallback download failed for %s: %r", url, e)

    return False

# ------------ TM загрузка CSV ------------
async def _tm_fetch_rows_csv(sheet_id: str, gid: str, sheet_name: str, override_url: str = "") -> list[list[str]]:
    import csv, io as _io
    urls = []
    if override_url:
        urls.append(override_url)
    if sheet_id:
        urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}")
        urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/pub?gid={gid}&single=true&output=csv")
        if sheet_name:
            from urllib.parse import quote
            urls.append(f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={quote(sheet_name)}")
    last_err = None
    for url in urls:
        try:
            resp = await asyncio.to_thread(requests.get, url, timeout=30)
            status = resp.status_code
            ctype = resp.headers.get("Content-Type", "")
            content = resp.content.decode("utf-8", errors="replace")
            if status == 200 and ("," in content or ";" in content or "\n" in content):
                reader = csv.reader(_io.StringIO(content))
                rows = [row for row in reader]
                if rows and any(any(cell.strip() for cell in row) for row in rows):
                    return rows
            last_err = f"Bad content from {url} (status={status}, type={ctype})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} at {url}"
            continue
    raise RuntimeError(last_err or "CSV fetch failed")

async def tm_load_data() -> list[list[str]]:
    if not TM_ENABLE:
        return []
    try:
        return await _tm_fetch_rows_csv(TM_SHEET_ID, TM_SHEET_GID, TM_SHEET_NAME, TM_SHEET_CSV_URL)
    except Exception as e:
        logger.error("TM: не удалось загрузить данные листа: %s", repr(e))
        if TM_DEBUG:
            raise
        return []

def _row_matches_registered(row: list[str]) -> bool:
    col = (row[5] if len(row) > 5 else "") or ""
    return "регистрация" in col.lower()

def _row_matches_expertise(row: list[str]) -> bool:
    col = (row[5] if len(row) > 5 else "") or ""
    return "экспертиза" in col.lower()

def _row_matches_keywords(row: list[str], keywords: list[str]) -> bool:
    low = [str(c or "").lower() for c in row]
    for kw in keywords:
        kw = kw.strip().lower()
        if kw and any(kw in cell for cell in low):
            return True
    return False

async def tm_process_search(chat_id: int, condition_cb, context: ContextTypes.DEFAULT_TYPE):
    try:
        data = await tm_load_data()
    except Exception as e:
        msg = "Данные листа недоступны или пусты."
        if TM_DEBUG:
            msg += f"\n\nДиагностика: {e!r}\nПроверьте публикацию таблицы (File→Publish to web), верный GID листа и доступность CSV-экспорта."
            msg += f"\nТекущие параметры: SHEET_ID={TM_SHEET_ID}, GID={TM_SHEET_GID}, SHEET_NAME={TM_SHEET_NAME}"
        await context.bot.send_message(chat_id, msg)
        return

    if not data or not any(data):
        note = "Данные листа недоступны или пусты."
        if TM_DEBUG:
            note += f"\nПроверьте: Publish to web включён, правильный GID, лист не пустой."
        await context.bot.send_message(chat_id, note)
        return

    rows = data[1:] if len(data) > 1 else []
    matched_idx = [i for i, r in enumerate(rows, start=2) if condition_cb(r)]
    if not matched_idx:
        await context.bot.send_message(chat_id, "Данные не найдены.")
        return

    for i in matched_idx:
        row = data[i-1]
        text, images = tm_format_row(row)
        if text:
            await context.bot.send_message(chat_id, text, parse_mode="HTML", disable_web_page_preview=False)
        for url in images:
            ok = await _tm_send_image_safely(chat_id, url, context)
            if not ok:
                logger.warning("TM: failed to send image after fallback: %s", url)

# ------------ TELEGRAM HANDLERS ------------
TM_MODE = "tm"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "docs"
    usage_left = "∞" if is_admin(update.effective_user.id) else max(
        0, DAILY_FREE_LIMIT - get_usage(update.effective_user.id)
    )
    msg = (
        "Привет!\n\n"
        "1) Пришлите файл (.pdf, .docx или .txt) и задайте вопрос по его содержанию (режим \"Вопросы по документам\").\n"
        "2) Нажмите \"🤖 AI-чат\" для диалога без документов.\n"
        "3) Нажмите \"🏷️ Товарные знаки\" для поиска по Google Sheets.\n\n"
        f"Сегодняшний лимит AI-чат: {usage_left} сообщений."
    )
    await update.message.reply_text(msg, reply_markup=MAIN_KB)

async def ai_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "ai"
    usage_left = "∞" if is_admin(update.effective_user.id) else max(0, DAILY_FREE_LIMIT - get_usage(update.effective_user.id))
    await update.message.reply_text(
        f"Режим: AI-чат. Спросите что угодно. Доступно сегодня: {usage_left}.", reply_markup=MAIN_KB
    )

async def docs_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "docs"
    await update.message.reply_text(
        "Режим: вопросы по проиндексированным документам. Пришлите файл и задавайте вопрос.", reply_markup=MAIN_KB
    )

async def tm_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = TM_MODE
    intro = (
        "Режим: 🏷️ Товарные знаки.\n\n"
        "Отправьте название/ключевые слова — найду строки в Google Sheets и пришлю карточки.\n"
        "Команды:\n"
        "• /tm_reg — записи, где статус содержит «регистрация»\n"
        "• /tm_exp — записи, где статус содержит «экспертиза»"
    )
    await update.message.reply_text(intro, reply_markup=MAIN_KB)

async def tm_cmd_reg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await tm_process_search(update.effective_chat.id, _row_matches_registered, context)

async def tm_cmd_exp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await tm_process_search(update.effective_chat.id, _row_matches_expertise, context)

async def tm_handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    kws = re.split(r"\s+", user_text)
    await tm_process_search(update.effective_chat.id, lambda row: _row_matches_keywords(row, kws), context)

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document: Document = update.message.document
    raw = document.file_name or "file"
    fname = Path(raw).name
    base, ext = os.path.splitext(fname)
    ext = ext.lower().lstrip(".")
    if ext not in ("pdf", "docx", "txt"):
        await update.message.reply_text("Поддерживаются только PDF, DOCX, TXT.")
        return
    os.makedirs(DOC_FOLDER, exist_ok=True)
    ts = int(time.time())
    file_path = os.path.join(DOC_FOLDER, f"{base}_{ts}.{ext}")
    new_file = await context.bot.get_file(document.file_id)
    await new_file.download_to_drive(file_path)

    manifest = load_manifest()
    file_hash = sha256_file(file_path)
    hashes = manifest.get("hashes", {})
    if file_hash in hashes:
        await update.message.reply_text("Этот файл уже проиндексирован ранее. Можете задавать вопросы.")
        return

    await update.message.reply_text("Файл загружен. Индексация началась...")
    def _index_job():
        try:
            added, total = index_file(file_path)
            return (True, added, total, None)
        except Exception as e:
            return (False, 0, 0, repr(e))
    ok, added, total, err = await asyncio.to_thread(_index_job)
    if ok:
        if added == 0:
            await update.message.reply_text(
                "Файл загружен, но текст не извлечён. Возможно, это скан без текстового слоя.\n"
                "Пришлите DOCX/TXT или PDF с текстом, либо включите OCR (tesseract+poppler/Docker)."
            )
            return
        manifest.setdefault("hashes", {})[file_hash] = {"fname": os.path.basename(file_path), "time": int(time.time())}
        save_manifest(manifest)
        await update.message.reply_text(
            f"Индексация завершена. Добавлено фрагментов: {added}. Всего: {total}. Теперь можно задавать вопросы."
        )
    else:
        await update.message.reply_text(f"❌ Ошибка индексации: {err}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_query = (update.message.text or "").strip()
    mode = context.user_data.get("mode", "docs")

    if user_query == AI_LABEL:
        await ai_mode(update, context); return
    if user_query == DOCS_LABEL:
        await docs_mode(update, context); return
    if user_query == TM_LABEL:
        await tm_mode(update, context); return

    if mode == "ai":
        uid = update.effective_user.id
        if not is_admin(uid):
            used = get_usage(uid)
            if used >= DAILY_FREE_LIMIT:
                await update.message.reply_text(
                    "Достигнут дневной лимит бесплатных обращений к ИИ (10). Возвращайтесь завтра или обратитесь к администратору.",
                    reply_markup=MAIN_KB,
                )
                return
        def _ai_job():
            try:
                return generate_direct_ai_answer(user_query)
            except Exception as e:
                return f"⚠️ Ошибка при обработке запроса: {repr(e)}"
        answer = await asyncio.to_thread(_ai_job)
        if answer and not is_admin(uid):
            inc_usage(uid)
        await send_long(update, answer)
        if not is_admin(uid):
            left = max(0, DAILY_FREE_LIMIT - get_usage(uid))
            await update.message.reply_text(f"Остаток на сегодня: {left}.")
        return

    if mode == "tm":
        low = user_query.lower()
        if low.startswith("/tm_reg"):
            await tm_cmd_reg(update, context); return
        if low.startswith("/tm_exp"):
            await tm_cmd_exp(update, context); return
        await tm_handle_text(update, context)
        return

    if not (os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE)):
        await update.message.reply_text("Нет проиндексированных документов. Сначала загрузите файл.")
        return

    def _answer_job():
        try:
            chunks = retrieve_chunks(user_query, k=RETRIEVAL_K)
            return generate_answer_with_gemini(user_query, chunks)
        except Exception as e:
            return f"⚠️ Ошибка при обработке запроса: {repr(e)}"

    answer = await asyncio.to_thread(_answer_job)
    if not answer:
        await update.message.reply_text("⚠️ Пустой ответ.")
    else:
        await send_long(update, answer)

async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    def stat():
        def fs(path):
            exists = os.path.exists(path)
            size = Path(path).stat().st_size if exists and os.path.isfile(path) else 0
            return f"{path}: {'OK' if exists else '—'} (size={size})"
        lines = [
            fs(DOC_FOLDER),
            fs(INDEX_FILE),
            fs(TEXTS_FILE),
            fs(MANIFEST_FILE),
            fs(USAGE_FILE),
            f"ADMIN_USER_IDS={sorted(list(ADMIN_USER_IDS))}",
            f"DAILY_FREE_LIMIT={DAILY_FREE_LIMIT}",
        ]
        try:
            texts = load_texts()
            lines.append(f"texts count={len(texts) if isinstance(texts, list) else 0}")
        except Exception as e:
            lines.append(f"texts load error: {e!r}")
        try:
            idx = load_index()
            lines.append(f"faiss ntotal={getattr(idx, 'ntotal', 0) if idx else 0}")
        except Exception as e:
            lines.append(f"faiss load error: {e!r}")
        try:
            data = _load_usage()
            today = _today_str()
            today_sum = sum(int(v.get(today, 0)) for v in data.values())
            lines.append(f"AI-чат обращений сегодня: {today_sum}")
        except Exception as e:
            lines.append(f"usage load error: {e!r}")
        return "\n".join(lines)
    out = await asyncio.to_thread(stat)
    await update.message.reply_text("Состояние:\n" + out)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error while processing update: %s", update)

# ------------ FASTAPI + PTB APP ------------
app = FastAPI()

@app.head("/")
async def health_head():
    return Response(status_code=200)

@app.get("/")
async def health_get():
    return {"ok": True}

application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("debug", debug))
application.add_handler(CommandHandler("ai", ai_mode))
application.add_handler(CommandHandler("docs", docs_mode))
application.add_handler(CommandHandler("tm", tm_mode))
application.add_handler(CommandHandler("tm_reg", tm_cmd_reg))
application.add_handler(CommandHandler("tm_exp", tm_cmd_exp))
application.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(AI_LABEL)}$"), ai_mode))
application.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(DOCS_LABEL)}$"), docs_mode))
application.add_handler(MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(TM_LABEL)}$"), tm_mode))
application.add_handler(MessageHandler(filters.Document.ALL, handle_file))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
application.add_error_handler(error_handler)

@app.on_event("startup")
async def _startup():
    await application.initialize()
    if RUN_MODE == "polling":
        await application.start()
        logger.info("PTB polling started")
    elif RUN_MODE == "webhook":
        if PUBLIC_BASE_URL:
            try:
                wh_url = f"{PUBLIC_BASE_URL}/telegram/{TELEGRAM_BOT_TOKEN}"
                await application.bot.set_webhook(url=wh_url, drop_pending_updates=True)
                logger.info("Webhook set to %s", wh_url)
            except Exception as e:
                logger.warning("Failed to set webhook: %r", e)
        logger.info("PTB initialized for webhook mode")
    logger.info("PTB initialized")

@app.on_event("shutdown")
async def _shutdown():
    if RUN_MODE == "polling":
        await application.stop()
        logger.info("PTB polling stopped")
    await application.shutdown()
    logger.info("PTB shutdown complete")

@app.post(f"/telegram/{TELEGRAM_BOT_TOKEN}")
async def telegram_webhook_token(request: Request):
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}
