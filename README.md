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
2. Initialize a Python 3.8+ virtual environment for development. `transcap`is the name of the venv, you can name anything you prefer.
```bash
python -m venv transcap
source transcap/bin/activate
```
3. Install neccessary Python packages.
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install pillow mss pytesseract langdetect
```
Note: Translation is built-in using Google Translate's free API (no package or API key needed!). For production use with higher reliability, consider using DeepL or Google Cloud Translation API.

4. (Optional) A translation API key (DeepL / Google Cloud Translation / Microsoft Translator) for more reliable translation. Otherwise, the code uses the free Google Translate API by default (no setup required).

## Usage

### Basic usage
Run the script with the region coordinates for the captions area:
```bash
python live_trans_terminal.py --left 300 --top 880 --width 1100 --height 160 --fps 2
```

### Using the run script
Or use the provided run script (edit `run.sh` to adjust the region coordinates):
```bash
chmod +x run.sh
./run.sh
```

### Finding the right coordinates
To find the correct screen coordinates for your captions:
1. Take a screenshot and open it in Preview (or similar)
2. Hover over the top-left corner of the caption area and note the X,Y coordinates
3. Measure the width and height of the caption region
4. Use these values for `--left`, `--top`, `--width`, and `--height`

### Command-line options
- `--left`: X coordinate of the left edge of the capture region (required)
- `--top`: Y coordinate of the top edge of the capture region (required)
- `--width`: Width of the capture region in pixels (required)
- `--height`: Height of the capture region in pixels (required)
- `--fps`: Capture rate (default: 1.5). Higher = more responsive but more CPU usage
- `--min-len`: Minimum text length to process (default: 2)
- `--duplicate-ttl`: Seconds before repeating same OCR line (default: 5.0)

Press Ctrl-C to stop the translator.