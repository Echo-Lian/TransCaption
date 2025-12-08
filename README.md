# TransCaption
TransCaption is a real-time translation tool for Teams meetings, supporting **two modes**:

## Modes

### OCR Mode (Original)
Captures a fixed screen region (where Teams captions appear), runs OCR continuously, detects language (Chinese/English), and translates to the other language. Displays translations in the terminal.

### Speech Mode (New!)
Captures system audio from Teams meetings, transcribes speech using OpenAI Whisper (runs locally, no API key needed), detects language (Chinese/English), and translates to the other language. Ideal for better accuracy compared to OCR.

Both modes:
- Auto-detect language per line/utterance
- Translate Chinese ↔ English
- Display results in terminal with timestamps
- Deduplicate repeated text

## Pipeline

### OCR Mode Pipeline
1. Repeatedly screenshot a small rectangular region (the captions area)
2. Run OCR (Tesseract) to extract text
3. Normalize & deduplicate (avoid repeating same lines)
4. For each new line: detect language (Chinese vs English)
5. Translate to the other language (Chinese → English or English → Chinese)
6. Print translated text to terminal

### Speech Mode Pipeline
1. Capture system audio via BlackHole (virtual audio device)
2. Transcribe audio with Whisper (local speech recognition)
3. Detect language per utterance (Chinese vs English)
4. Translate to the other language
5. Print translated text to terminal with fuzzy deduplication

## Requirements (for macOS)

### For Both Modes
1. Initialize a Python 3.8+ virtual environment for development. `transcap` is the name of the venv, you can name anything you prefer.
```bash
python -m venv transcap
source transcap/bin/activate
```

2. Install necessary Python packages.
```bash
pip install -r requirements.txt
```

Note: Translation is built-in using Google Translate's free API (no package or API key needed!). For production use with higher reliability, consider using DeepL or Google Cloud Translation API.

### Additional Requirements for OCR Mode
1. Install Tesseract OCR (system binary):
```bash
brew install tesseract
```

### Additional Requirements for Speech Mode
1. Install FFmpeg (required by Whisper):
```bash
brew install ffmpeg
```

2. Install BlackHole (virtual audio device for capturing system audio):
```bash
brew install blackhole-2ch
```

3. **Configure Audio MIDI Setup** (one-time setup):
   - Open **Audio MIDI Setup** (Applications > Utilities > Audio MIDI Setup)
   - Click the **"+"** button at the bottom-left and select **"Create Multi-Output Device"**
   - In the Multi-Output Device, check both:
     - **BlackHole 2ch** (for capturing)
     - **Your speakers** (e.g., "MacBook Pro Speakers" - so you can hear the audio)
   - Right-click the Multi-Output Device and select **"Use This Device For Sound Output"**
   - This routes audio to both your speakers (so you hear it) and BlackHole (so the program can capture it)

4. On first run, Whisper will download the model (~500MB for "small" model). This is a one-time download.

## Usage

### Speech Mode (Recommended for Accuracy)

**Basic usage:**
```bash
python live_trans_terminal.py --mode speech
```

This will:
- Capture system audio from BlackHole
- Transcribe speech using Whisper (small model by default)
- Translate Chinese ↔ English automatically
- Display translations in terminal

**Advanced options:**
```bash
# Use a smaller/faster model (less accurate but faster)
python live_trans_terminal.py --mode speech --model-size base

# Use a larger/more accurate model (slower but better accuracy)
python live_trans_terminal.py --mode speech --model-size medium

# Customize audio chunk duration (trade-off between latency and accuracy)
python live_trans_terminal.py --mode speech --chunk-duration 5.0
```

**Speech mode options:**
- `--model-size`: Whisper model size: `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
  - `tiny`: Fast but less accurate
  - `base`: Faster, acceptable accuracy
  - `small`: **Recommended** - good balance
  - `medium`: More accurate, slower
  - `large`: Most accurate, slowest
- `--chunk-duration`: Audio chunk duration in seconds (default: 3.0)
- `--sample-rate`: Audio sample rate (default: 16000)
- `--device-name`: Audio device name (default: "BlackHole 2ch")
- `--duplicate-ttl`: Seconds before repeating same text (default: 5.0)

### OCR Mode (Original)

**Basic usage:**
Run the script with the region coordinates for the captions area:
```bash
python live_trans_terminal.py --mode ocr --left 300 --top 880 --width 1100 --height 160 --fps 2
```

**Using the run script:**
Or use the provided run script (edit `run.sh` to adjust the region coordinates):
```bash
chmod +x run.sh
./run.sh
```

**Finding the right coordinates:**
To find the correct screen coordinates for your captions:
1. Take a screenshot and open it in Preview (or similar)
2. Hover over the top-left corner of the caption area and note the X,Y coordinates
3. Measure the width and height of the caption region
4. Use these values for `--left`, `--top`, `--width`, and `--height`

**OCR mode options:**
- `--left`: X coordinate of the left edge of the capture region (required)
- `--top`: Y coordinate of the top edge of the capture region (required)
- `--width`: Width of the capture region in pixels (required)
- `--height`: Height of the capture region in pixels (required)
- `--fps`: Capture rate (default: 1.5). Higher = more responsive but more CPU usage
- `--min-len`: Minimum text length to process (default: 2)
- `--duplicate-ttl`: Seconds before repeating same OCR line (default: 5.0)

### General Notes
- Press **Ctrl-C** to stop the translator
- First run of speech mode will download the Whisper model (~500MB for "small")
- Speech mode provides better accuracy than OCR, especially for Chinese characters
- Expected latency for speech mode: 5-8 seconds

## Troubleshooting

### Speech Mode Issues

**"Audio device 'BlackHole 2ch' not found"**
- Install BlackHole: `brew install blackhole-2ch`
- Verify installation: Open Audio MIDI Setup and check if BlackHole 2ch appears in the device list
- If using a different device, specify it with `--device-name`

**"No audio being captured" or "No output"**
- Ensure Audio MIDI Setup is configured correctly (see Requirements section)
- Make sure the Multi-Output Device is set as the system default output
- Test by playing audio - you should hear it through your speakers
- Check that BlackHole is selected in the Multi-Output Device

**"Model download fails" or "FFmpeg error"**
- Install FFmpeg: `brew install ffmpeg`
- Check internet connection for first-time model download
- Try a smaller model: `--model-size base`

**"Transcription is too slow"**
- Use a smaller model: `--model-size base` or `--model-size tiny`
- Default `small` model works well on most modern Macs
- Close other CPU-intensive applications

**"Too many duplicate translations"**
- Increase `--duplicate-ttl` (e.g., `--duplicate-ttl 10.0`)
- The fuzzy matching threshold can be adjusted in the code (`FUZZY_MATCH_THRESHOLD`)

### OCR Mode Issues

**"Tesseract not found"**
- Install Tesseract: `brew install tesseract`
- If installed in a custom location, set `TESSERACT_CMD` in the script

**"OCR accuracy is poor"**
- Ensure the caption region is captured correctly (adjust coordinates)
- Make sure captions are large enough and clearly visible
- Consider using Speech mode for better accuracy