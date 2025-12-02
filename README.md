# TransCaption
TransCaption captures a small fixed region of the screen (mainly used in where Teams captions show), runs OCR continuously, automatically detects whether the line is Chinese or English (there will be more language pairs in the feature), translates into the other language, and displays an always-on-top overlay showing the translation. It can handle mixed-language captions (it detects language per line).

## Pipeline
1. Repeatedly screenshot a small rectangular region (the captions area).
2. Run OCR (Tesseract) to extract text.
3. Normalize & deduplicate (avoid repeating same lines).
4. For each new line: detect language (simple language detection).
5. Send to translation API (DeepL / Google Translate / Microsoft Translator).
6. Show translated text in an always-on-top overlay (or print to console).

## Requirements (for macOS)
1. Tesseract OCR installed (system binary).
```bash
brew install tesseract
```
2. Initialize a Python 3.8+ virtual environment for development.
```bash
python -m venv path/to/venv/
```
3. Install neccessary Python packages.
```bash
pip install pillow mss pytesseract langdetect requests googletrans
```
4. A translation API key (DeepL / Google Cloud Translation / Microsoft Translator) OR use free unofficial `googletrans` (less reliable).