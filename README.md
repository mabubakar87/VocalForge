# VocalForge

VocalForge is a Python-based application designed for real-time audio recording, transcription, and text formatting. It leverages the **Faster Whisper** model for accurate and fast transcription, and provides a user-friendly GUI built with **Tkinter**. The application supports both live audio recording and uploading of pre-recorded audio files for transcription. The transcribed text is automatically formatted and copied to the clipboard for easy use.

---

## Features

- **Real-Time Audio Recording**: Record audio directly from your microphone.
- **Audio Transcription**: Transcribe recorded or uploaded audio files using the Faster Whisper model.
- **Text Formatting**: Automatically format transcribed text for better readability (e.g., capitalization, punctuation spacing).
- **Clipboard Integration**: Copy the transcribed text to the clipboard for easy pasting into other applications.
- **Upload Audio Files**: Transcribe pre-recorded `.wav` files.
- **CUDA Support**: Utilize GPU acceleration (if available) for faster transcription.
- **Auto-Hiding Scrollbar**: The scrollbar in the text area hides when not needed, providing a cleaner UI.
- **Logging**: Detailed logging for debugging and tracking application activity.

---

## Requirements

- Python 3.x
- Libraries:
  - `tkinter`
  - `sounddevice`
  - `numpy`
  - `queue`
  - `threading`
  - `faster-whisper`
  - `pyautogui`
  - `pyperclip`
  - `wave`
  - `re`
  - `subprocess`
  - `os`
  - `sys`
  - `logging`
  - `datetime`
  - `requests`

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mabubakar87/VocalForge.git
   cd VocalForge
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

---

## Usage

1. **Recording Audio**:
   - Click the red circle button to start recording.
   - Click the button again to stop recording and begin transcription.

2. **Uploading Audio Files**:
   - Click the "Upload Audio File" button to select a `.wav` file for transcription.

3. **Viewing Transcription**:
   - The transcribed text will appear in the text box below the buttons.

4. **Copying Text**:
   - The transcribed text is automatically copied to the clipboard and can be pasted into any application.

---

## Logging

The application logs all activities to a file named `superwhisper.log` in the current working directory. Logs include timestamps, log levels, and messages.

---

## Notes

- **Internet Connection**: An internet connection is required for the first-time model download.
- **CUDA Support**: If a CUDA-compatible GPU is available, the application will automatically use it for faster transcription.
- **Audio Format**: Uploaded audio files must be in `.wav` format (mono, 16-bit, 16kHz).

---

## Troubleshooting

- **CUDA Not Available**: If CUDA is not available, the application will fall back to CPU processing, which may be slower.
- **Audio Recording Issues**: Ensure that your microphone is properly configured and accessible by the application.
- **Transcription Errors**: Check the log file for detailed error messages and ensure that the audio file is in the correct format.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Faster Whisper**: The transcription model used in this application.
- **Tkinter**: The GUI framework used for building the interface.
- **Sounddevice**: The library used for audio recording.

---

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact the maintainer directly.

---

Enjoy using **VocalForge**! 🎙️✨