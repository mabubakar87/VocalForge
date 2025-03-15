# VocalForge

VocalForge is a voice transcription application that allows users to record or upload audio files and transcribe them into text using the Faster Whisper model.

Requirements

Python 3.8+

pip

Dependencies listed in requirements.txt

Installation

Clone the repository:

git clone https://github.com/mabubakar87/VocalForge.git
cd VocalForge

Install dependencies:

pip install -r requirements.txt

Running the Application

To start the application, run:

python main.py

Creating an Executable

To create a standalone executable:

Install PyInstaller:

pip install pyinstaller

Generate the executable:

pyinstaller --onefile --windowed main.py

The executable will be available in the dist/ folder.

How to Use

Recording Audio: Click the record button to start and stop recording.

Uploading an Audio File: Click the "Upload Audio File" button and select a .wav file.

Transcription: The transcribed text will be displayed in the text area and copied to the clipboard automatically.

Logging

Logs are stored in superwhisper.log and include details about application usage and errors.