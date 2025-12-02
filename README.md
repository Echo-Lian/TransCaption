# TransCaption
TransCaption captures a small fixed region of the screen (mainly used in where Teams captions show), runs OCR continuously, automatically detects whether the line is Chinese or English (there will be more language pairs in the feature!), translates into the other language, and displays an always-on-top overlay showing the translation. It can handle mixed-language captions (it detects language per line).

## Pipeline
### Repeatedly screenshot a small rectangular region (the captions area).

### Run OCR (Tesseract) to extract text.

### Normalize & deduplicate (avoid repeating same lines).

### For each new line: detect language (simple language detection).

### Send to translation API (DeepL / Google Translate / Microsoft Translator).

### Show translated text in an always-on-top overlay (or print to console).

## Installation

```bash
pip install ...
```