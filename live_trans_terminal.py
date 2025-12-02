#!/usr/bin/env python3
"""
live_trans_terminal.py
Terminal-only live caption translator:
- Screenshot a fixed region repeatedly
- OCR with Tesseract (chi_sim+eng)
- Detect language per line (CJK check + langdetect)
- Translate to the other language
- Print new translations to terminal, de-duplicated with TTL

Usage:
    python live_trans_terminal.py --left 300 --top 880 --width 1100 --height 160 --fps 2

Adjust the region to match your Teams captions.
"""

import argparse
import time
import sys
import re
import json
from collections import deque
from datetime import datetime
from urllib import parse, request
from urllib.error import URLError, HTTPError

import mss
from PIL import Image
import pytesseract
from langdetect import detect, DetectorFactory

# translator choices
USE_GOOGLETRANS = True   # default easy option (no API key)
USE_DEEPL = False
USE_GOOGLE_CLOUD = False

# If tesseract is not on PATH, set full path here, e.g. "/opt/homebrew/bin/tesseract"
TESSERACT_CMD = None  # set if needed, e.g. "/opt/homebrew/bin/tesseract"

# DeepL / Google Cloud config placeholders (if you switch to those options)
DEEPL_AUTH_KEY = None  # e.g. "YOUR_DEEPL_KEY"
GOOGLE_CLOUD_KEY = None  # e.g. "YOUR_GOOGLE_CLOUD_KEY"

# Setup deterministic langdetect
DetectorFactory.seed = 0

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------- translation backends ----------------
def translate_googletrans(text, dest):
    """
    Free Google Translate using direct HTTP requests.
    No API key needed, no external package required.
    """
    if not text or not text.strip():
        return text

    # Google Translate expects 'zh-CN' not 'zh-cn'
    if dest == 'zh-cn':
        dest = 'zh-CN'

    try:
        # URL encode the text
        encoded_text = parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={dest}&dt=t&q={encoded_text}"

        # Create request with headers to mimic browser
        req = request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        # Make request
        with request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))

        # Parse response - result is nested arrays
        # Format: [[[translated_text, original_text, null, null, ...], ...], ...]
        if result and len(result) > 0 and result[0]:
            translated_parts = [part[0] for part in result[0] if part[0]]
            return ''.join(translated_parts)

        return text

    except (URLError, HTTPError, json.JSONDecodeError, IndexError, KeyError) as e:
        # If translation fails, return original text
        raise RuntimeError(f"Translation failed: {e}")

def translate_deepl(text, dest):
    # deepl python package
    try:
        import deepl
    except ImportError:
        raise RuntimeError("deepl is not installed. Run: pip install deepl")
    auth_key = DEEPL_AUTH_KEY or ""
    if not auth_key:
        raise RuntimeError("DEEPL_AUTH_KEY not set")
    translator = deepl.Translator(auth_key)
    # DeepL target codes: EN, ZH
    target = "EN" if dest == "en" else "ZH"
    res = translator.translate_text(text, target_lang=target)
    return res.text

def translate_google_cloud(text, dest):
    # google cloud translate: short example
    try:
        from google.cloud import translate_v2 as translate
    except ImportError:
        raise RuntimeError(
            "google-cloud-translate is not installed. Run: pip install google-cloud-translate"
        )
    client = translate.Client(api_key=GOOGLE_CLOUD_KEY) if GOOGLE_CLOUD_KEY else translate.Client()
    result = client.translate(text, target_language=dest)
    return result["translatedText"]

def translate_text(text, dest):
    if USE_GOOGLETRANS:
        return translate_googletrans(text, dest)
    if USE_DEEPL:
        return translate_deepl(text, dest)
    if USE_GOOGLE_CLOUD:
        return translate_google_cloud(text, dest)
    raise RuntimeError("No translator configured")

# ---------------- OCR and language detection ----------------
def ocr_image(pil_img):
    # Use both Chinese and English tesseract models if installed
    config = r'-c tessedit_char_blacklist=¦‘†‡°·• --psm 6'
    try:
        text = pytesseract.image_to_string(pil_img, lang='chi_sim+eng', config=config)
    except Exception:
        text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
    return text.strip()

def has_cjk(s):
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def decide_target_lang(s):
    """
    Decide target translation language based on source text.
    Returns 'en' if source is Chinese, 'zh' if source is English, None if undetermined.
    """
    s = s.strip()
    if not s:
        return None

    # If contains CJK characters -> it's Chinese/zh -> translate to English
    if has_cjk(s):
        return "en"

    # Try language detection for non-CJK text
    try:
        lang = detect(s)
        # If detected as English or English-like, translate to Chinese
        if lang.startswith("en"):
            return "zh-cn"
        # For other languages (e.g., Spanish, French), default to translating to English
        # This is a reasonable fallback for mixed/international captions
        return "en"
    except Exception:
        # If detection fails (e.g., too short text), assume English -> translate to Chinese
        # This is a safe default for Latin-script text
        return "zh-cn"

# ---------------- main loop ----------------
def main(args):
    region = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    fps = args.fps
    min_len = args.min_len
    duplicate_ttl = args.duplicate_ttl

    sct = mss.mss()
    seen = {}  # text -> last_seen_time

    print("Starting live terminal translator.")
    print(f"Region: {region}, fps: {fps}")
    print("Press Ctrl-C to stop.\n")

    try:
        while True:
            start = time.time()
            shot = sct.grab(region)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            text = ocr_image(img)
            now = time.time()

            if not text:
                # nothing
                time.sleep(max(0, 1.0/fps - (time.time()-start)))
                continue

            # break into lines and process
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in lines:
                if len(ln) < min_len:
                    continue
                last = seen.get(ln, 0)
                if now - last < duplicate_ttl:
                    continue
                seen[ln] = now

                target = decide_target_lang(ln)
                if not target:
                    continue
                # translate
                try:
                    translated = translate_text(ln, target)
                except Exception as e:
                    translated = f"[translate error: {e}]"

                ts = datetime.now().strftime("%H:%M:%S")
                if target == "en":
                    tag = "ZH → EN"
                elif target in ["zh", "zh-cn"]:
                    tag = "EN → ZH"
                else:
                    tag = f"→ {target.upper()}"
                # print original and translated in a compact form
                print(f"[{ts}] {tag}")
                print(f"  {ln}")
                print(f"  → {translated}\n")
                # small sleep to avoid hitting translator too fast
                time.sleep(0.12)

            # housekeeping: drop old seen entries occasionally
            if len(seen) > 2000:
                cutoff = now - (duplicate_ttl * 2)
                for k in list(seen.keys()):
                    if seen[k] < cutoff:
                        del seen[k]

            elapsed = time.time() - start
            to_sleep = max(0, 1.0/fps - elapsed)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--top", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--duplicate-ttl", type=float, default=5.0,
                   help="seconds before repeating same OCR line")
    args = p.parse_args()
    main(args)
