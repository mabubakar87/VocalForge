import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import pyautogui
import pyperclip
import wave
import re
import subprocess
import os
import sys
import logging
from datetime import datetime
import requests

# Set up logging to a file with timestamps
log_file = os.path.join(os.getcwd(), "superwhisper.log")  # Log file in the current working directory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler()  # Also log to the console (optional)
    ]
)

logging.info("Application started.")

def is_cuda_available():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"CUDA check failed: {e}")
        return False

def is_internet_available():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

device = "cuda" if is_cuda_available() else "cpu"
logging.info(f"Using device: {device}")

try:
    model = WhisperModel("distil-small.en", device=device)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    if not is_internet_available():
        messagebox.showerror(
            "Internet Required",
            "Internet is required for first-time model downloading. Please connect to the internet and try again."
        )
        sys.exit(1)
    else:
        raise

audio_queue = queue.Queue()
recording = False
processing = False

def callback(indata, frames, time, status):
    if status:
        logging.warning(f"Audio input status: {status}")
    audio_queue.put(indata.copy())

def record_audio():
    global recording
    recording = True
    audio_data = []
    try:
        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            while recording:
                audio_data.append(audio_queue.get())
        return np.concatenate(audio_data, axis=0)
    except Exception as e:
        logging.error(f"Error during audio recording: {e}")
        raise

def save_wav(filename, data, samplerate=16000):
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())
        logging.info(f"Audio saved to {filename}.")
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")
        raise

def toggle_recording():
    global recording, processing
    if not recording:
        text_output.delete(1.0, tk.END)
        status_label.config(text="Recording...", fg="red")
        canvas.itemconfig(record_button, fill="#F44336")
        threading.Thread(target=record).start()
    else:
        recording = False
        status_label.config(text="Processing...", fg="white")
        processing = True
        canvas.itemconfig(record_button, fill="#4CAF50")

def record():
    global recording, processing
    try:
        audio_data = record_audio()
        save_wav("recorded_audio.wav", audio_data)
        transcribe_audio("recorded_audio.wav")        
        processing = False
    except Exception as e:
        logging.error(f"Error during recording or transcription: {e}")
        status_label.config(text="Error", fg="red")

def format_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,!?])', r'\1', text)    
    text = re.sub(r'(?<=[.!?])\s*(\w)', lambda m: m.group(1).upper(), text)
    text = text[0].upper() + text[1:] if text else text
    text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)
    return text

def transcribe_audio(file_path):
    try:
        segments, _ = model.transcribe(file_path)
        text = " ".join(segment.text for segment in segments)
        formatted_text = format_text(text)
        insert_text(formatted_text)
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, formatted_text)
        logging.info("Transcription completed successfully.")
		status_label.config(text="Transcription Completed", fg="white")
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def insert_text(text):
    try:
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        logging.info("Text copied to clipboard.")
    except Exception as e:
        logging.error(f"Error copying text to clipboard: {e}")

def upload_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        status_label.config(text="Transcribing...", fg="white")
        threading.Thread(target=lambda: transcribe_audio(file_path)).start()        
		
root = tk.Tk()
root.title("SuperWhisper GUI")
root.geometry("500x450")
root.configure(bg="#2E2E2E")
root.resizable(True, True)

frame = tk.Frame(root, bg="#2E2E2E")
frame.pack(pady=20)

status_label = tk.Label(frame, text="Idle", fg="white", bg="#2E2E2E", font=("Arial", 12))
status_label.pack(pady=5)

canvas = tk.Canvas(frame, width=80, height=80, bg="#2E2E2E", highlightthickness=0)
canvas.pack(pady=10)

record_button = canvas.create_oval(10, 10, 70, 70, fill="#4CAF50", outline="")
canvas.create_oval(25, 25, 55, 55, fill="white", outline="")
canvas.bind("<Button-1>", lambda event: toggle_recording())

upload_button = tk.Button(root, text="Upload Audio File", command=upload_audio, bg="#555", fg="white")
upload_button.pack(pady=10)

text_output = scrolledtext.ScrolledText(root, height=5, width=50, wrap=tk.WORD, bg="#1E1E1E", fg="white", font=("Arial", 12))
text_output.pack(pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
