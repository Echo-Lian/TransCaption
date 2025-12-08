#!/usr/bin/env python3
"""
live_trans_terminal.py
Terminal-only live caption translator:

OCR Mode:
- Screenshot a fixed region repeatedly
- OCR with Tesseract (chi_sim+eng)
- Detect language per line (CJK check + langdetect)
- Translate to the other language
- Print new translations to terminal, de-duplicated with TTL

Speech Mode:
- Capture system audio (via BlackHole on macOS)
- Transcribe with Whisper (local, supports Chinese + English)
- Detect language and translate to the other language
- Print translations to terminal, de-duplicated with TTL

Usage:
    OCR Mode:
        python live_trans_terminal.py --mode ocr --left 300 --top 880 --width 1100 --height 160 --fps 2

    Speech Mode:
        python live_trans_terminal.py --mode speech

Adjust the region to match your Teams captions (OCR mode).
For speech mode, ensure BlackHole is installed and configured.
"""

import argparse
import time
import sys
import re
import json
import queue
import difflib
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

# Speech recognition constants
DEFAULT_SAMPLE_RATE = 16000  # Whisper's native sample rate
DEFAULT_CHUNK_DURATION = 3.0  # Audio chunk size in seconds (balance between latency and accuracy)
DEFAULT_MODEL_SIZE = "small"  # Whisper model size: tiny, base, small, medium, large
DEFAULT_DEVICE_NAME = "BlackHole 2ch"  # Default audio device for macOS
SPEECH_MIN_LENGTH = 5  # Minimum transcribed text length to process
FUZZY_MATCH_THRESHOLD = 0.80  # 80% similarity threshold for deduplication

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

# ---------------- speech recognition functions ----------------
def get_blackhole_device_index(device_name=DEFAULT_DEVICE_NAME):
    """
    Find audio input device by name and return device index.
    Raises RuntimeError if device not found.
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "sounddevice is not installed. Run: pip install sounddevice\n"
            "Also ensure system dependencies are installed: brew install portaudio"
        )

    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device_name.lower() in device['name'].lower() and device['max_input_channels'] > 0:
            return i

    # If not found, show available devices
    print("\nAvailable audio input devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")

    raise RuntimeError(
        f"\nAudio device '{device_name}' not found.\n"
        f"Please install BlackHole: brew install blackhole-2ch\n"
        f"Or specify a different device with --device-name"
    )

def is_similar_text(text1, text2, threshold=FUZZY_MATCH_THRESHOLD):
    """
    Check if two texts are similar using fuzzy matching.
    Returns True if similarity ratio >= threshold.
    """
    ratio = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio >= threshold

def speech_main(args):
    """
    Speech recognition mode - capture system audio and translate in real-time.
    """
    try:
        import sounddevice as sd
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"\nError: Missing dependency for speech mode: {e}")
        print("\nPlease install speech recognition dependencies:")
        print("  pip install faster-whisper sounddevice numpy")
        print("\nAlso ensure system dependencies are installed:")
        print("  brew install ffmpeg blackhole-2ch")
        sys.exit(1)

    # Load Whisper model
    print(f"Loading Whisper '{args.model_size}' model...")
    print("(First run will download the model, ~500MB for 'small' model)")
    try:
        model = WhisperModel(args.model_size, device="cpu", compute_type="int8")
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"\nError loading Whisper model: {e}")
        print("Make sure ffmpeg is installed: brew install ffmpeg")
        sys.exit(1)

    # Find audio device
    try:
        device_idx = get_blackhole_device_index(args.device_name)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)

    # Setup audio streaming
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        """Callback function for sounddevice to capture audio chunks."""
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    # Calculate chunk sizes
    chunk_size_samples = int(args.sample_rate * args.chunk_duration)
    overlap_samples = int(args.sample_rate * 0.5)  # 0.5 second overlap

    # Deduplication dictionary with fuzzy matching
    seen = {}  # text -> last_seen_time
    seen_texts = []  # List of recent texts for fuzzy comparison

    print(f"Listening to audio device: {args.device_name}")
    print(f"Sample rate: {args.sample_rate} Hz, Chunk duration: {args.chunk_duration}s")
    print("Press Ctrl-C to stop.\n")

    try:
        with sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=args.sample_rate,
            callback=audio_callback,
            blocksize=4096
        ):
            chunk_accumulator = []

            while True:
                # Accumulate audio until we have enough for a chunk
                while len(chunk_accumulator) < chunk_size_samples:
                    try:
                        data = audio_queue.get(timeout=0.1)
                        chunk_accumulator.extend(data.flatten())
                    except queue.Empty:
                        continue

                # Process chunk
                audio_chunk = np.array(chunk_accumulator[:chunk_size_samples], dtype=np.float32)
                # Keep overlap for next chunk
                chunk_accumulator = chunk_accumulator[chunk_size_samples - overlap_samples:]

                # Transcribe with Whisper
                try:
                    segments, info = model.transcribe(
                        audio_chunk,
                        language=None,  # Auto-detect language
                        task="transcribe",
                        vad_filter=True,  # Voice activity detection to skip silence
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )

                    # Process each segment
                    for segment in segments:
                        text = segment.text.strip()

                        if len(text) < SPEECH_MIN_LENGTH:
                            continue

                        # Check for duplicates with fuzzy matching
                        now = time.time()
                        is_duplicate = False

                        # Check exact match first
                        if text in seen:
                            last = seen[text]
                            if now - last < args.duplicate_ttl:
                                is_duplicate = True

                        # Check fuzzy match against recent texts
                        if not is_duplicate:
                            for prev_text in seen_texts[-10:]:  # Check last 10 texts
                                if is_similar_text(text, prev_text):
                                    if prev_text in seen and now - seen[prev_text] < args.duplicate_ttl:
                                        is_duplicate = True
                                        break

                        if is_duplicate:
                            continue

                        # Update seen dictionary and list
                        seen[text] = now
                        seen_texts.append(text)
                        if len(seen_texts) > 50:  # Keep only recent 50 texts
                            seen_texts.pop(0)

                        # Detect target language
                        target = decide_target_lang(text)
                        if not target:
                            continue

                        # Translate
                        try:
                            translated = translate_text(text, target)
                        except Exception as e:
                            translated = f"[translate error: {e}]"

                        # Print with timestamp
                        ts = datetime.now().strftime("%H:%M:%S")
                        if target == "en":
                            tag = "[SPEECH] ZH → EN"
                        elif target in ["zh", "zh-cn"]:
                            tag = "[SPEECH] EN → ZH"
                        else:
                            tag = f"[SPEECH] → {target.upper()}"

                        print(f"[{ts}] {tag}")
                        print(f"  {text}")
                        print(f"  → {translated}\n")

                        # Small sleep to avoid hitting translator too fast
                        time.sleep(0.12)

                except Exception as e:
                    print(f"Transcription error: {e}", file=sys.stderr)
                    continue

                # Housekeeping: drop old seen entries
                if len(seen) > 2000:
                    cutoff = now - (args.duplicate_ttl * 2)
                    for k in list(seen.keys()):
                        if seen[k] < cutoff:
                            del seen[k]

    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)

# ---------------- OCR mode main loop ----------------
def ocr_main(args):
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

def main():
    """
    Main dispatcher - routes to OCR or speech mode based on --mode argument.
    """
    p = argparse.ArgumentParser(
        description="TransCaption: Live translation for Teams meetings (OCR or Speech mode)"
    )

    # Mode selection (required)
    p.add_argument(
        "--mode",
        choices=["ocr", "speech"],
        required=True,
        help="Mode: 'ocr' for screen capture + OCR, 'speech' for audio capture + speech recognition"
    )

    # OCR mode arguments
    ocr_group = p.add_argument_group("OCR mode arguments")
    ocr_group.add_argument("--left", type=int, help="X coordinate of capture region (required for OCR mode)")
    ocr_group.add_argument("--top", type=int, help="Y coordinate of capture region (required for OCR mode)")
    ocr_group.add_argument("--width", type=int, help="Width of capture region (required for OCR mode)")
    ocr_group.add_argument("--height", type=int, help="Height of capture region (required for OCR mode)")
    ocr_group.add_argument("--fps", type=float, default=1.5, help="Capture rate (default: 1.5)")
    ocr_group.add_argument("--min-len", type=int, default=2, help="Minimum text length to process (default: 2)")

    # Speech mode arguments
    speech_group = p.add_argument_group("Speech mode arguments")
    speech_group.add_argument(
        "--model-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE})"
    )
    speech_group.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Audio chunk duration in seconds (default: {DEFAULT_CHUNK_DURATION})"
    )
    speech_group.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate (default: {DEFAULT_SAMPLE_RATE})"
    )
    speech_group.add_argument(
        "--device-name",
        type=str,
        default=DEFAULT_DEVICE_NAME,
        help=f"Audio device name (default: '{DEFAULT_DEVICE_NAME}')"
    )

    # Common arguments
    p.add_argument(
        "--duplicate-ttl",
        type=float,
        default=5.0,
        help="Seconds before repeating same text (default: 5.0)"
    )

    args = p.parse_args()

    # Validate mode-specific requirements
    if args.mode == "ocr":
        if not all([args.left is not None, args.top is not None,
                    args.width is not None, args.height is not None]):
            p.error("OCR mode requires: --left, --top, --width, --height")
        ocr_main(args)
    elif args.mode == "speech":
        speech_main(args)
    else:
        p.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
