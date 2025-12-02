"""
Live screen-caption translator:
- continuously OCRs a screen region,
- detects Chinese vs English per line,
- translates to the opposite language,
- shows latest result in an always-on-top small window.

Dependencies:
    pip install pillow mss pytesseract langdetect requests
System:
    Install Tesseract OCR:
      - Windows: https://github.com/UB-Mannheim/tesseract
      - macOS: brew install tesseract
      - Linux: apt install tesseract-ocr
"""

import time
import threading
from collections import deque
from datetime import datetime

import mss
from PIL import Image
import pytesseract
from langdetect import detect, DetectorFactory
import requests
import tkinter as tk

DetectorFactory.seed = 0  # deterministic language detection

# -------------------- CONFIG --------------------
REGION = {"left": 300, "top": 900, "width": 1000, "height": 180}
# Adjust REGION to match your Teams caption box coordinates (pixels)

FPS = 2.0  # how many captures per second
MIN_LEN = 1  # ignore tiny OCR noise
DUPLICATE_TTL = 6.0  # seconds before repeating same text

# Choose translator:
USE_DEEPL = False
DEEPL_API_KEY = "YOUR_DEEPL_API_KEY"  # if using DeepL
USE_GOOGLE_CLOUD = False
GOOGLE_CLOUD_KEY = "YOUR_GOOGLE_CLOUD_KEY"  # if using Google Cloud Translate
USE_GOOGLETRANS_UNOFFICIAL = True  # fallback (pip install googletrans==4.0.0-rc1)

# ------------------------------------------------

# simple cache to avoid repeating same translation
seen_cache = {}  # text -> timestamp

# translation functions
def translate_deepl(text, target_lang):
    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = {"text": text, "target_lang": target_lang}
    r = requests.post(url, data=data, headers=headers, timeout=10)
    r.raise_for_status()
    j = r.json()
    return j["translations"][0]["text"]

def translate_google_cloud(text, target_lang):
    url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_CLOUD_KEY}"
    data = {"q": text, "target": target_lang}
    r = requests.post(url, json=data, timeout=10)
    r.raise_for_status()
    return r.json()["data"]["translations"][0]["translatedText"]

def translate_googletrans(text, target_lang):
    # unofficial googletrans: language code examples: 'en', 'zh-CN'
    from googletrans import Translator
    t = Translator()
    return t.translate(text, dest=target_lang).text

def translate_text(text, target_lang):
    if USE_DEEPL:
        return translate_deepl(text, target_lang)
    if USE_GOOGLE_CLOUD:
        return translate_google_cloud(text, target_lang)
    if USE_GOOGLETRANS_UNOFFICIAL:
        return translate_googletrans(text, target_lang)
    raise RuntimeError("No translator configured")

# ----------------- OCR & detect -----------------
def ocr_image(pil_img):
    # Tesseract config - improve OCR for mixed Chinese/English:
    custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\u4e00-\u9fff，。！？：；“”\'\"-()[]{}—\n --psm 6'
    text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng', config=custom_config)
    return text.strip()

def decide_target_lang(text):
    # treat mostly Chinese characters as zh, else en
    # attempt langdetect but be conservative
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    # langdetect returns 'zh-cn' or 'zh-tw' sometimes; map to 'zh'
    if lang.startswith("zh"):
        return "en"
    if lang.startswith("en"):
        return "zh"
    # fallback: check for presence of CJK characters
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "en"
    # default: translate to Chinese
    return "zh"

# ---------------- overlay UI --------------------
class Overlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)  # no border
        self.root.attributes("-alpha", 0.85)
        self.label = tk.Label(self.root, text="", font=("Helvetica", 18), justify="left")
        self.label.pack(padx=6, pady=6)
        # place top-right by default
        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"+{screen_w-520}+60")
        # click to close
        self.root.bind("<Button-3>", lambda e: self.root.quit())

    def update_text(self, text):
        # show timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        self.label.config(text=f"{ts}  {text}")

    def mainloop(self):
        self.root.mainloop()

# ---------------- main loop --------------------
def capture_loop(overlay):
    sct = mss.mss()
    last_text = ""
    last_time = 0
    while True:
        start = time.time()
        img = sct.grab(REGION)
        pil = Image.frombytes("RGB", img.size, img.rgb)
        text = ocr_image(pil)
        if not text or len(text) <= MIN_LEN:
            # nothing useful
            time.sleep(max(0, 1.0/FPS - (time.time()-start)))
            continue

        # dedupe: ignore if identical to last OCR within TTL
        now = time.time()
        if text == last_text and (now - last_time) < DUPLICATE_TTL:
            time.sleep(max(0, 1.0/FPS - (time.time()-start)))
            continue
        last_text = text
        last_time = now

        # split into lines and handle each
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            # skip short noise
            if len(ln) < 2:
                continue
            if ln in seen_cache and (now - seen_cache[ln]) < DUPLICATE_TTL:
                continue
            seen_cache[ln] = now

            target = decide_target_lang(ln)
            # mapping for translation API language codes
            if target == "zh":
                target_code = "ZH"
                # DeepL expects ZH or ZH for Chinese? For Google use 'zh'
                target_code_gc = "zh"
            else:
                target_code = "EN"
                target_code_gc = "en"

            try:
                if USE_DEEPL:
                    translated = translate_text(ln, target_code)
                elif USE_GOOGLE_CLOUD:
                    translated = translate_text(ln, target_code_gc)
                else:
                    translated = translate_text(ln, target_code_gc)
            except Exception as e:
                translated = f"[translate error] {e}"
            display = f"{ln}\n→ {translated}"
            overlay.update_text(display)
            # small pause to allow user to read
            time.sleep(0.35)

        # pacing
        elapsed = time.time() - start
        sleep_for = max(0, 1.0/FPS - elapsed)
        time.sleep(sleep_for)

def start():
    overlay = Overlay()
    t = threading.Thread(target=capture_loop, args=(overlay,), daemon=True)
    t.start()
    overlay.mainloop()

if __name__ == "__main__":
    start()
