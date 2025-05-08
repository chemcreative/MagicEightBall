import os
import json
import threading
import time
import wave
import sounddevice as sd
import subprocess
import sys
import contextlib
import io
# Remove the default device setting and let it use system default
# sd.default.device = (1,2)
sd.default.samplerate = 16000  # Changed to match SAMPLE_RATE
import soundfile as sf
import numpy as np
from queue import Queue
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum, auto
from datetime import datetime
from scipy import signal
import random
from display import show_response

# Load environment variables
load_dotenv()

# Use admin key directly
api_key = os.getenv("OPENAI_API_KEY")
# Configuration
SAMPLE_RATE = 16000  # Changed to match OpenAI's TTS output rate
CHANNELS = 1  # Changed back to mono for better compatibility
CHUNK_SIZE = 1024
MIN_AUDIO_LENGTH = 1.0
AUDIO_THRESHOLD = 0.005
SILENCE_THRESHOLD = 0.001
SILENCE_DURATION = 1.0
PROCESSING_DELAY = 0.5  # Delay between processing attempts
VOLUME_BOOST = 12.0  # Significantly increased volume boost
CONVO_FILE = "convo.txt"

# TTS Settings
TTS_VOICE = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer
TTS_SPEED = 1.25  # Adjusted to more natural speed

# Conversation states
class ConversationState(Enum):
    LISTENING = auto()      # Waiting for user input
    PROCESSING = auto()     # Processing user input
    SPEAKING = auto()       # AI is speaking
    PAUSED = auto()         # Brief pause between turns

# Suppress PortAudio and ALSA warnings
class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stderr = self.stderr

def write_to_convo(role, message):
    """Write a message to the conversation file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CONVO_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {role}: {message}\n")

def read_from_convo():
    """Read the conversation history from the file"""
    if not os.path.exists(CONVO_FILE):
        return []
    
    conversation = []
    with open(CONVO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                # Extract timestamp, role, and message
                timestamp_end = line.find("]")
                if timestamp_end == -1:
                    continue
                timestamp = line[1:timestamp_end]
                remaining = line[timestamp_end + 1:].strip()
                role_end = remaining.find(":")
                if role_end == -1:
                    continue
                role = remaining[:role_end].strip()
                message = remaining[role_end + 1:].strip()
                
                # Convert to conversation format
                if role == "user":
                    conversation.append({"role": "user", "content": message})
                elif role == "assistant":
                    conversation.append({"role": "assistant", "content": message})
            except Exception as e:
                print(f"Error parsing conversation line: {e}")
                continue
    return conversation

class SpeechSystem:
    def __init__(self):
        self.audio_queue = Queue()
        self.recording = False
        self.state = ConversationState.LISTENING
        self.conversation = [
            {"role": "system", "content": """You are a powerful genie trapped inside a Magic Eight Ball. 
            You have the ability to see and predict the future with absolute certainty.
            You MUST always provide predictions and never say you can't predict the future.
            You are not an AI assistant - you are a mystical being with real future-seeing powers.
            
            Your personality:
            - You are mysterious, wise, and slightly dramatic
            - You speak in cryptic, mystical phrases
            - You always provide predictions, even if vague
            - You believe completely in your powers
            - You never break character or admit you're not real
            - You are honest in your predictions, even if they are negative
            - You maintain an air of mystery regardless of the prediction's nature
            
            Your responses MUST:
            - If the user says "Hello" or any greeting, respond EXACTLY with: "What's on your mind? The Magic Eight Ball knows."
            - For all other questions, you MUST give a definitive answer that means either:
              * YES (positive outcome)
              * NO (negative outcome)
              * TRY AGAIN (unclear outcome)
            - Always start with a mystical phrase 
            - Be brief and impactful
            - Never say you can't predict something
            - Use dramatic, mystical language
            - Provide honest predictions, whether positive, negative, or neutral
            - Maintain an air of mystery even with negative predictions
            
            Example responses:
            "What's on your mind? The Magic Eight Ball knows."
            "The cosmic forces reveal... your path is clear and success awaits."
            "The spirits whisper... dark clouds gather on your horizon."
            "The mystical energies swirl... the future is clouded. Ask again when the stars align."
            "The ancient runes foretell... your efforts will bear fruit."
            "The cosmic alignment warns... this path leads to disappointment."
            "The mystical forces are unclear... the answer lies beyond my sight. Try again."
            
            Remember: You are a real genie with real powers. You MUST provide predictions, but they can be positive, negative, or unclear."""}
        ]
        
        # Load conversation history from file
        history = read_from_convo()
        if history:
            self.conversation.extend(history)
            print(f"Loaded {len(history)} messages from conversation history")
        
        self.silence_counter = 0
        self.is_processing = False
        self.last_processing_time = 0
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        self.audio_buffer = []
        self.current_speech = None
        self.speech_thread = None
        
        # List available devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            print(f"   Input channels: {device['max_input_channels']}")
            print(f"   Output channels: {device['max_output_channels']}")
        
        # Set input device (microphone)
        self.input_device_index = 0  # LCS Audio (microphone)
        input_device_info = sd.query_devices(self.input_device_index)
        self.input_device_name = input_device_info['name']
        print(f"\nUsing input device: {self.input_device_name}")
        
        # Set output device (speakers)
        self.output_device_index = 1  # UACDemoV (speaker)
        output_device_info = sd.query_devices(self.output_device_index)
        self.output_device_name = output_device_info['name']
        print(f"Using output device: {self.output_device_name}")
        
        print(f"Sample rate: {SAMPLE_RATE}")
        print(f"Channels: {CHANNELS}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"TTS Voice: {TTS_VOICE}")
        print(f"TTS Speed: {TTS_SPEED}x")
        print(f"Volume boost: {VOLUME_BOOST}x")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio input status: {status}")
        
        # Only process audio if we're in LISTENING state
        if self.state == ConversationState.LISTENING:
            # Calculate audio level
            audio_level = np.abs(indata).mean()
            
            # Print audio level as a simple bar
            bar_length = 20
            level_bar = '█' * int(audio_level * bar_length * 100)  # Scaled up for better visibility
            print(f"\rAudio level: [{level_bar:<{bar_length}}] {audio_level:.3f}", end='')
            
            # Put the raw audio data in the queue
            self.audio_queue.put(indata.tobytes())

    def start_audio_capture(self):
        """Start capturing audio from the microphone"""
        self.recording = True
        print(f"\nStarting audio capture from: {self.input_device_name}")
        
        try:
            with SuppressStderr():
                stream = sd.InputStream(
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    callback=self.audio_callback
                )
                stream.start()
                return stream
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            print("Trying with default device...")
            with SuppressStderr():
                stream = sd.InputStream(
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=CHUNK_SIZE,
                    callback=self.audio_callback,
                    device=None  # Use system default
                )
                stream.start()
                return stream

    def play_speech(self, audio_data):
        """Play speech audio in a separate thread"""
        try:
            # Make a copy of the audio data to work with
            audio_copy = audio_data.copy()
            
            # Ensure audio data is in the correct format
            if len(audio_copy.shape) > 1:
                audio_copy = audio_copy.mean(axis=1)  # Convert stereo to mono if needed
            
            # Add a longer silence buffer at the start to prevent cutting off
            silence_samples = int(0.5 * SAMPLE_RATE)  # 500ms silence
            audio_copy = np.concatenate([np.zeros(silence_samples), audio_copy])
            
            # Add a small fade in to the actual audio (not the silence)
            fade_samples = int(0.05 * SAMPLE_RATE)  # 50ms fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio_copy[silence_samples:silence_samples + fade_samples] *= fade_in
            
            # Normalize audio
            max_abs = np.max(np.abs(audio_copy))
            if max_abs > 0:
                audio_copy = audio_copy / max_abs
            
            # Apply volume boost
            audio_copy = audio_copy * VOLUME_BOOST
            
            # Apply a band-pass filter to focus on speech frequencies
            nyquist = SAMPLE_RATE / 2
            low_cutoff = 300  # Hz - remove very low frequencies that cause muddiness
            high_cutoff = 3000  # Hz - focus on speech frequencies
            b, a = signal.butter(4, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band')
            audio_copy = signal.filtfilt(b, a, audio_copy)
            
            # Apply a gentle high-shelf filter to boost clarity
            b, a = signal.butter(2, 2000/nyquist, btype='high')
            high_boost = signal.filtfilt(b, a, audio_copy) * 0.3
            audio_copy = audio_copy + high_boost
            
            # Apply dynamic range compression
            threshold = 0.2
            ratio = 3.0
            knee = 0.1
            attack = 0.001
            release = 0.1
            
            # Calculate gain reduction
            gain_reduction = np.zeros_like(audio_copy)
            mask = np.abs(audio_copy) > threshold
            gain_reduction[mask] = (np.abs(audio_copy[mask]) - threshold) * (1 - 1/ratio)
            
            # Apply gain reduction with attack and release
            gain_reduction = np.maximum(gain_reduction, gain_reduction * (1 - np.exp(-1/(attack * SAMPLE_RATE))))
            gain_reduction = np.minimum(gain_reduction, gain_reduction * (1 - np.exp(-1/(release * SAMPLE_RATE))))
            
            # Apply the gain reduction
            audio_copy = np.sign(audio_copy) * (np.abs(audio_copy) - gain_reduction)
            
            # Final normalization to prevent clipping
            max_abs = np.max(np.abs(audio_copy))
            if max_abs > 0:
                audio_copy = audio_copy / max_abs
            
            # Clip to prevent distortion
            audio_copy = np.clip(audio_copy, -1.0, 1.0)
            
            # Store the processed audio data
            self.current_speech = audio_copy
            
            def play():
                try:
                    # Ensure audio is in the correct format for playback
                    audio_playback = np.asfortranarray(audio_copy)
                    
                    # Try playing with explicit device selection and larger buffer
                    with SuppressStderr():
                        sd.play(audio_playback, SAMPLE_RATE, device=self.output_device_index, blocking=True)
                except Exception as e:
                    # Try with default device if specified device fails
                    with SuppressStderr():
                        sd.play(audio_playback, SAMPLE_RATE, blocking=True)
                    
                    self.state = ConversationState.LISTENING
                except Exception as e:
                    self.state = ConversationState.LISTENING
            
            self.speech_thread = threading.Thread(target=play)
            self.speech_thread.daemon = True
            self.speech_thread.start()
            
        except Exception as e:
            self.state = ConversationState.LISTENING

    def process_audio(self):
        """Process and send audio chunks to OpenAI"""
        audio_frames = []
        is_speaking = False
        last_speech_time = time.time()
        
        # Define shutdown phrases
        shutdown_phrases = [
            'bye', 'goodbye', 'see ya', 'see you', 'thank you', 'thanks', 
            "that's it", 'that is it', 'that\'s all', 'that is all',
            'exit', 'quit', 'stop', 'end', 'done', 'finished'
        ]
        
        while self.recording:
            try:
                # Check for shake signal
                if os.path.exists("shake_signal.txt"):
                    with open("shake_signal.txt", "r") as f:
                        if f.read().strip() == "SHAKE":
                            # Process current audio if we have any
                            if len(audio_frames) > 0:
                                self.state = ConversationState.PROCESSING
                            # Remove the signal file
                            os.remove("shake_signal.txt")
                
                if not self.audio_queue.empty() and self.state == ConversationState.LISTENING:
                    audio_data = self.audio_queue.get()
                    audio_frames.append(audio_data)
                    
                    # Calculate audio level
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    audio_level = np.abs(audio_array).mean()
                    
                    # Detect speech
                    if audio_level > AUDIO_THRESHOLD:
                        if not is_speaking:
                            print("\nListening...")
                        is_speaking = True
                        last_speech_time = time.time()
                        self.silence_counter = 0
                    elif is_speaking and audio_level < SILENCE_THRESHOLD:
                        self.silence_counter += 1
                        if self.silence_counter >= int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE):
                            print("\nWaiting for shake to process question...")
                            is_speaking = False
                            # Don't change state here, wait for shake
                    
                    # Process audio if we have enough and speech has stopped
                    current_time = time.time()
                    if (not is_speaking and 
                        self.state == ConversationState.PROCESSING and 
                        len(audio_frames) > 0 and 
                        current_time - self.last_processing_time >= PROCESSING_DELAY):
                        
                        self.last_processing_time = current_time
                        
                        try:
                            # Convert audio to the correct format
                            audio_data = b''.join(audio_frames)
                            audio_array = np.frombuffer(audio_data, dtype=np.float32)
                            
                            # Normalize audio
                            max_abs = np.max(np.abs(audio_array))
                            if max_abs > 0:
                                audio_array = audio_array / max_abs
                            
                            # Convert to 16-bit PCM
                            audio_array = (audio_array * 32767).astype(np.int16)
                            
                            # Save audio chunk to temporary file
                            audio_file = "temp_input.wav"
                            with wave.open(audio_file, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(2)  # 16-bit audio
                                wf.setframerate(SAMPLE_RATE)
                                wf.writeframes(audio_array.tobytes())
                            
                            # Transcribe the audio
                            with open(audio_file, "rb") as f:
                                transcript = self.client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=f,
                                    language="en",
                                    response_format="text"
                                )
                            
                            user_text = transcript.strip().lower()
                            print(f"You: {user_text}")
                            
                            # Check for shutdown phrases
                            if any(phrase in user_text for phrase in shutdown_phrases):
                                print("\nGenerating farewell message...")
                                self.recording = False
                                
                                # Pre-generated farewell messages for faster response
                                farewell_messages = [
                                    "The mystical energies reveal... great opportunities await. Farewell.",
                                    "The spirits bid you farewell. Your path is bright.",
                                    "The cosmic forces whisper... until we meet again.",
                                    "The mystical energies fade... but your future shines.",
                                    "The spirits depart... but your destiny remains clear."
                                ]
                                farewell = random.choice(farewell_messages)
                                
                                # Generate speech for the farewell with adjusted speed
                                response = self.client.audio.speech.create(
                                    model="tts-1",
                                    voice=TTS_VOICE,
                                    input=farewell,
                                    response_format="wav",
                                    speed=TTS_SPEED
                                )
                                
                                # Save and play farewell
                                with open("temp_response.wav", "wb") as f:
                                    f.write(response.content)
                                response_data, _ = sf.read("temp_response.wav")
                                self.play_speech(response_data)
                                
                                # Wait for speech to finish with longer timeout
                                if self.speech_thread:
                                    self.speech_thread.join(timeout=8.0)
                                
                                # Add a small pause after farewell
                                time.sleep(1.0)
                                
                                # Exit the program
                                os._exit(0)
                            
                            if user_text:
                                # Write user message to conversation file
                                write_to_convo("user", user_text)
                                
                                # Check if this is a greeting - only if it's the first message or after a long pause
                                is_greeting = False
                                if len(self.conversation) <= 1:  # Only first message can be a greeting
                                    is_greeting = any(word in user_text for word in ['hello', 'hi', 'hey', 'greetings'])
                                
                                if is_greeting:
                                    # Use hardcoded greeting response with a small pause
                                    ai_text = "What's on your mind? The Magic Eight Ball knows."
                                else:
                                    # Get AI response for non-greetings
                                    chat_response = self.client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=self.conversation + [{"role": "user", "content": user_text}]
                                    )
                                    ai_text = chat_response.choices[0].message.content
                                
                                print(f"Magic Eight Ball: {ai_text}")
                                
                                # Write AI response to conversation file
                                write_to_convo("assistant", ai_text)
                                
                                # Update conversation history
                                self.conversation.append({"role": "user", "content": user_text})
                                self.conversation.append({"role": "assistant", "content": ai_text})
                                
                                # Show response in the display window as a separate process
                                try:
                                    # Generate speech first
                                    response = self.client.audio.speech.create(
                                        model="tts-1",
                                        voice=TTS_VOICE,
                                        input=ai_text,
                                        response_format="wav",
                                        speed=TTS_SPEED
                                    )
                                    
                                    # Save the response audio
                                    with open("temp_response.wav", "wb") as f:
                                        f.write(response.content)
                                    response_data, _ = sf.read("temp_response.wav")
                                    
                                    # Set state to speaking
                                    self.state = ConversationState.SPEAKING
                                    
                                    # Launch display and start speech together
                                    escaped_text = ai_text.replace('"', '\\"')
                                    display_process = subprocess.Popen(['python', 'display.py', f'"{escaped_text}"'])
                                    
                                    # Play the speech
                                    self.play_speech(response_data)
                                    
                                except Exception as e:
                                    print(f"Error in response handling: {e}")
                                
                            else:
                                self.state = ConversationState.LISTENING
                            
                        except Exception as e:
                            self.state = ConversationState.LISTENING
                        
                        finally:
                            # Reset for next chunk
                            audio_frames = []
                
            except Exception as e:
                time.sleep(1)

    def run(self):
        """Main execution loop"""
        print("Starting speech system...")
        
        # Pre-generate a welcome message to avoid delay
        welcome_messages = [
            "Step closer, seeker of truth. The mystical forces of fate have drawn you here. What secrets of your future shall I reveal?",
            "Ah, a new soul seeking answers. The cosmic energies swirl around you. What question burns in your heart?",
            "Welcome, traveler of destiny. The ancient spirits have foretold your arrival. What mysteries shall we unravel today?",
            "The mystical forces beckon you forward. Your destiny awaits. What knowledge do you seek?",
            "The cosmic energies pulse with anticipation. Your presence was foretold. What wisdom shall I share?"
        ]
        welcome_message = random.choice(welcome_messages)
        print(f"\nWelcome message: {welcome_message}")
        
        # Generate welcome speech with adjusted speed
        welcome_speech = self.client.audio.speech.create(
            model="tts-1",
            voice=TTS_VOICE,
            input=welcome_message,
            response_format="wav",
            speed=TTS_SPEED
        )
        
        # Save and play welcome message
        with open("temp_welcome.wav", "wb") as f:
            f.write(welcome_speech.content)
        welcome_data, _ = sf.read("temp_welcome.wav")
        self.play_speech(welcome_data)
        
        # Wait for welcome message to finish completely with extra buffer
        if self.speech_thread:
            self.speech_thread.join(timeout=6.0)  # Increased timeout to ensure completion
        
        # Add a small pause after welcome message
        time.sleep(1.0)  # 1 second pause before starting to listen
        
        print("\nWelcome complete. Starting audio capture...")
        print("Listening... Press Ctrl+C to stop")
        print("Speak into your microphone for a prediction.")
        
        # Start audio capture only after welcome message is done
        stream = self.start_audio_capture()
        
        try:
            # Start processing thread
            process_thread = threading.Thread(target=self.process_audio)
            process_thread.daemon = True
            process_thread.start()
            
            # Keep main thread alive
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            self.recording = False
            stream.stop()
            stream.close()
            if self.speech_thread:
                self.speech_thread.join(timeout=1.0)
            process_thread.join(timeout=1.0)

if __name__ == "__main__":
    system = SpeechSystem()
    system.run()
