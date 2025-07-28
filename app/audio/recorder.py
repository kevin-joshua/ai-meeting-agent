# meeting_recorder.py
import pyaudio
import wave
import time
import threading
import signal
import sys
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class MeetingRecorder:
    """Enhanced recorder for meetings with manual start/stop control."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.output_file = None
        self.frames = []
        self.is_recording = False
        self.audio_thread = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.device_index = None
        self.start_time = None
        self.recording_duration = 0
        
        # Create recordings directory if it doesn't exist
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n🛑 Stopping recording gracefully...")
        self.stop_recording()
        self.cleanup()
        print("👋 Goodbye! Your meeting has been saved.")
        sys.exit(0)

    def get_input_devices(self) -> List[Dict[str, Any]]:
        """List all input audio devices."""
        devices = []
        for i in range(self.pyaudio_instance.get_device_count()):
            try:
                info = self.pyaudio_instance.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': info['defaultSampleRate']
                    })
            except Exception as e:
                print(f"⚠️  Warning: Could not get info for device {i}: {e}")
        return devices

    def find_voicemeeter_aux_output(self) -> Optional[int]:
        """Find the index of VoiceMeeter AUX Output."""
        devices = self.get_input_devices()
        for device in devices:
            device_name = device['name'].lower()
            # Check for various VoiceMeeter device names
            if any(vm_name in device_name for vm_name in [
                "voicemeeter out b2", "voicemeeter aux output", 
                "voicemeeter output", "vm-vaio3"
            ]):
                return device['index']
        return None

    def list_devices(self):
        """Display all available input devices."""
        print("\n🎧 Available Audio Input Devices:")
        print("=" * 50)
        devices = self.get_input_devices()
        
        voicemeeter_found = False
        for device in devices:
            device_name = device['name'].lower()
            is_voicemeeter = any(vm_name in device_name for vm_name in [
                "voicemeeter", "vm-vaio"
            ])
            
            status = " ⭐ RECOMMENDED FOR MEETINGS" if is_voicemeeter else ""
            if is_voicemeeter:
                voicemeeter_found = True
                
            print(f"{device['index']:2d}: {device['name']}{status}")
            print(f"    Channels: {device['channels']}, Sample Rate: {int(device['sample_rate'])} Hz")
        
        print("=" * 50)
        if voicemeeter_found:
            print("✅ VoiceMeeter device found! This will capture both system audio and microphone.")
        else:
            print("⚠️  No VoiceMeeter device found. You may only capture microphone audio.")
            print("💡 Install VoiceMeeter for better meeting recordings: https://voicemeeter.com/")
        print()

    def select_device(self) -> Optional[int]:
        """Let user select an audio device."""
        self.list_devices()
        
        # Try to auto-select VoiceMeeter device
        vm_device = self.find_voicemeeter_aux_output()
        if vm_device is not None:
            print(f"🎯 Auto-selected VoiceMeeter device (Index: {vm_device})")
            response = input("Press Enter to use this device, or type device index to choose manually: ").strip()
            
            if response == "":
                return vm_device
            else:
                try:
                    return int(response)
                except ValueError:
                    print("❌ Invalid input. Using VoiceMeeter device.")
                    return vm_device
        else:
            # Manual selection
            while True:
                try:
                    device_index = input("Enter device index (or 'q' to quit): ").strip()
                    if device_index.lower() == 'q':
                        return None
                    
                    device_index = int(device_index)
                    devices = self.get_input_devices()
                    
                    if any(d['index'] == device_index for d in devices):
                        return device_index
                    else:
                        print(f"❌ Device index {device_index} not found. Please try again.")
                except ValueError:
                    print("❌ Please enter a valid number or 'q' to quit.")

    def generate_filename(self) -> str:
        """Generate a filename based on current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.recordings_dir, f"meeting_{timestamp}.wav")

    def start_recording(self, output_file: Optional[str] = None, device_index: Optional[int] = None):
        """Start recording from selected device."""
        if output_file is None:
            self.output_file = self.generate_filename()
        else:
            self.output_file = os.path.join(self.recordings_dir, output_file)
        
        if device_index is not None:
            self.device_index = device_index
        elif self.device_index is None:
            self.device_index = self.find_voicemeeter_aux_output()

        if self.device_index is None:
            print("❌ No audio device selected or found. Cannot start recording.")
            return False

        # Get device info for confirmation
        try:
            device_info = self.pyaudio_instance.get_device_info_by_index(self.device_index)
            print(f"🎙️  Recording from: {device_info['name']}")
        except Exception as e:
            print(f"⚠️  Warning: Could not get device info: {e}")

        print(f"📁 Output file: {self.output_file}")
        print(f"🔧 Settings: {self.sample_rate}Hz, {self.channels} channels")
        
        self.frames = []  # Clear any previous frames
        self.is_recording = True
        self.start_time = time.time()
        
        self.audio_thread = threading.Thread(target=self._record)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        print("✅ Recording started!")
        print("🛑 Press Ctrl+C to stop recording at any time")
        print("-" * 50)
        
        return True

    def _record(self):
        """Internal recording loop."""
        stream = None
        try:
            stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )

            frame_count = 0
            while self.is_recording:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                    
                    # Show progress every 5 seconds
                    frame_count += 1
                    if frame_count % (self.sample_rate // self.chunk_size * 5) == 0:
                        elapsed = time.time() - self.start_time
                        print(f"⏱️  Recording... {self._format_duration(elapsed)}")
                        
                except Exception as e:
                    print(f"⚠️  Warning during recording: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error starting recording stream: {e}")
            self.is_recording = False
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    print(f"⚠️  Warning closing stream: {e}")
                self._save_audio_file()

    def _format_duration(self, seconds: float) -> str:
        """Format duration in a readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def _save_audio_file(self):
        """Save recorded frames to a WAV file."""
        if not self.frames:
            print("⚠️  No audio data to save.")
            return
        
        try:
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            
            # Calculate file size and duration
            file_size = os.path.getsize(self.output_file)
            duration = len(self.frames) * self.chunk_size / self.sample_rate
            
            print(f"\n💾 Recording saved successfully!")
            print(f"📁 File: {self.output_file}")
            print(f"📊 Size: {file_size / (1024*1024):.1f} MB")
            print(f"⏱️  Duration: {self._format_duration(duration)}")
            
        except Exception as e:
            print(f"❌ Failed to save audio file: {e}")

    def stop_recording(self):
        """Stop the recording."""
        if not self.is_recording:
            print("⚠️  No recording in progress.")
            return
        
        print("\n🛑 Stopping recording...")
        self.is_recording = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=5.0)
        
        if self.start_time:
            total_duration = time.time() - self.start_time
            print(f"✅ Recording stopped after {self._format_duration(total_duration)}")

    def cleanup(self):
        """Clean up PyAudio resources."""
        self.stop_recording()
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                print(f"⚠️  Warning during cleanup: {e}")

    def monitor_recording(self):
        """Monitor the recording and provide status updates."""
        try:
            while self.is_recording:
                time.sleep(10)  # Update every 10 seconds
                if self.is_recording and self.start_time:
                    elapsed = time.time() - self.start_time
                    print(f"🎙️  Still recording... {self._format_duration(elapsed)} elapsed")
        except KeyboardInterrupt:
            # This will be handled by the signal handler
            pass


def main():
    """Main function to run the meeting recorder."""
    print("🤖 Doraemon's Meeting Recorder")
    print("=" * 40)
    print("📝 Perfect for recording meetings, calls, and presentations!")
    print("✨ Captures both system audio and microphone with VoiceMeeter")
    print()
    
    recorder = MeetingRecorder()
    
    try:
        # Device selection
        device_index = recorder.select_device()
        if device_index is None:
            print("👋 Recording cancelled.")
            return
        
        # Optional custom filename
        print("\n📁 File Naming:")
        custom_name = input("Enter custom filename with extension (.wav) (press Enter for auto-generated): ").strip()
        output_file = custom_name if custom_name else None
        
        # Start recording
        print("\n🚀 Starting recording...")
        if recorder.start_recording(output_file, device_index):
            # Keep the main thread alive and monitor
            recorder.monitor_recording()
        else:
            print("❌ Failed to start recording.")
            
    except KeyboardInterrupt:
        # This will be handled by the signal handler
        pass
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()