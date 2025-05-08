# 🎙️ OpenAI Speech-to-Speech on Raspberry Pi 3

This project demonstrates how to set up a real-time Speech-to-Speech system on a Raspberry Pi 3 using OpenAI's API.

## 🚀 Features
- Real-time speech capture using a USB microphone
- OpenAI API integration for speech processing
- Automatic streaming of audio chunks
- Console display of responses
- Small footprint, low-latency, Raspberry Pi optimized
- 100% Python 3, lightweight, no external server needed

## 🧰 Requirements
- Raspberry Pi 3 (or newer)
- USB microphone
- Speaker output (3.5mm jack or USB speaker)
- Python 3.11+
- OpenAI API key
- mpg123 (for audio playback)

## 🛠 Installation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install python3-pip python3-venv portaudio19-dev mpg123 -y

# Clone this repository
git clone https://github.com/chemcreative/MagicEightBall.git
cd MagicEightBall

# Create and activate virtual environment
python3 -m venv openai_env
source openai_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 🎯 Usage
1. Edit `.env` and add your OpenAI API Key
2. Activate the virtual environment:
   ```bash
   source openai_env/bin/activate
   ```
3. Run the application:
   ```bash
   python3 main.py
   ```

## 📚 About
This project is intended as a learning tool for developers interested in:
- Real-time AI voice interaction
- Low-resource device integrations
- Raspberry Pi creative technology projects

Built as a lightweight starter for more complex voice applications on embedded hardware.

## ⚠️ Note
Make sure your USB microphone and speaker are properly configured in your Raspberry Pi's audio settings before running the application. 