#!/bin/bash
source transcap/bin/activate

# TransCaption - Choose your mode:

# ===== SPEECH MODE (Recommended) =====
# Captures system audio and transcribes with Whisper
# Better accuracy than OCR, especially for Chinese
python live_trans_terminal.py --mode speech

# ===== OCR MODE (Original) =====
# Captures screen region and runs OCR
# Adjust coordinates to match your Teams caption area
# python live_trans_terminal.py --mode ocr --left 80 --top 700 --width 1120 --height 150 --fps 2

# ===== SPEECH MODE ADVANCED OPTIONS =====
# Use smaller/faster model
# python live_trans_terminal.py --mode speech --model-size base

# Use larger/more accurate model
# python live_trans_terminal.py --mode speech --model-size medium
